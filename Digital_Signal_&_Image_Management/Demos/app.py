import os
import io
import math
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable
from PIL import Image
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from dotenv import load_dotenv

load_dotenv()



# 1. SRRGAN CUSTOM LAYERS


@register_keras_serializable()
class SpatialAttentionBlock(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.concat = layers.Concatenate(axis=3)
        self.multiply = layers.Multiply()

    def build(self, input_shape):
        self.conv = layers.Conv2D(filters=1, kernel_size=7, padding='same', activation='sigmoid')
        super().build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=3, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=3, keepdims=True)
        concat = self.concat([avg_pool, max_pool])
        attention_map = self.conv(concat)
        return self.multiply([inputs, attention_map])


@register_keras_serializable()
class PixelShuffle(layers.Layer):
    def __init__(self, block_size, **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.block_size)

    def get_config(self):
        config = super().get_config()
        config.update({"block_size": self.block_size})
        return config



# 2. DEHAZING


def get_dark_channel(image, window_size=15):
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    return cv2.erode(min_channel, kernel)


def get_atmospheric_light(image, dark_channel):
    flat_image = image.reshape((-1, 3))
    flat_dark = dark_channel.ravel()
    num_top_pixels = int(max(flat_dark.size * 0.001, 1))
    indices = np.argpartition(flat_dark, -num_top_pixels)[-num_top_pixels:]
    candidates = flat_image[indices]
    return np.median(candidates, axis=0)


def get_transmission(image, A, omega=0.9, window_size=15):
    norm_image = image.astype(np.float64) / (A.astype(np.float64) + 1e-6)
    dc_norm = get_dark_channel(norm_image, window_size)
    transmission = 1 - omega * dc_norm
    brightness = np.max(image, axis=2) / 255.0
    protection = np.clip((brightness - 0.7) / (1.0 - 0.7), 0, 1)
    transmission = (1.0 - protection) * transmission + (protection * 1.0)
    return transmission


def refine_transmission(image, transmission_raw, r=60, eps=1e-4):
    gray_guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return cv2.ximgproc.guidedFilter(guide=gray_guide, src=transmission_raw.astype(np.float32), radius=r, eps=eps,
                                     dDepth=-1)


def apply_final_dehaze_enhancement(image, saturation_factor=1.05):
    img_min, img_max = image.min(), image.max()
    stretched = (image - img_min) / (img_max - img_min + 1e-6)
    img_gamma = np.power(stretched, 1.0 / 1.1)
    img_uint8 = (img_gamma * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    a_float = a.astype(np.float32)
    b_float = b.astype(np.float32)
    a_boosted = (a_float - 128) * saturation_factor + 128
    b_boosted = (b_float - 128) * saturation_factor + 128
    limg = cv2.merge((l, np.clip(a_boosted, 0, 255).astype(np.uint8), np.clip(b_boosted, 0, 255).astype(np.uint8)))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final.astype(np.float64) / 255.0


def run_dcp_pipeline(rgb_image):
    W = rgb_image.shape[1]
    scale = W / 600
    win_size = int(max(7, min(101, round(15 * scale))))
    if win_size % 2 == 0: win_size += 1

    guide_r = win_size * 4

    dc = get_dark_channel(rgb_image, win_size)
    A = get_atmospheric_light(rgb_image, dc)

    t_raw = get_transmission(rgb_image, A, 0.92, win_size)
    t_refined = refine_transmission(rgb_image, t_raw, guide_r, 1e-4)

    img_f = rgb_image.astype(np.float64) / 255.0
    A_f = A.astype(np.float64) / 255.0

    t_c = np.maximum(t_refined, 0.15)

    J = (img_f - A_f.reshape(1, 1, 3)) / np.expand_dims(t_c, axis=2) + A_f.reshape(1, 1, 3)
    J = np.clip(J, 0, 1)

    enhanced_float = apply_final_dehaze_enhancement(J, saturation_factor=1.05)

    final_uint8 = (enhanced_float * 255).astype(np.uint8)
    return final_uint8



# 3. CONFIG & MODEL LOADING


TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
PATCH_SIZE, UPSCALE_FACTOR, OVERLAP, BATCH_SIZE = 128, 2, 32, 8

print("Loading SRRGAN Model.")
model_sr = None
try:
    model_sr = tf.keras.models.load_model(
        "models/model_gan.keras",
        custom_objects={"SpatialAttentionBlock": SpatialAttentionBlock, "PixelShuffle": PixelShuffle},
        compile=False
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")



# 4. TILED SRRGAN LOGIC


async def run_sr_tiled(lr_image: Image.Image, context, chat_id, message_id):
    lr_array = np.array(lr_image.convert("RGB"), dtype=np.float32)
    lr_norm = (lr_array / 127.5) - 1.0
    h, w, _ = lr_norm.shape

    safety = 16
    lr_protected = np.pad(lr_norm, ((safety, safety), (safety, safety), (0, 0)), mode='reflect')
    ph_orig, pw_orig = lr_protected.shape[:2]

    stride = PATCH_SIZE - OVERLAP
    nx = math.ceil(max(0, pw_orig - PATCH_SIZE) / stride) + 1
    ny = math.ceil(max(0, ph_orig - PATCH_SIZE) / stride) + 1

    pad_h = (ny - 1) * stride + PATCH_SIZE - ph_orig
    pad_w = (nx - 1) * stride + PATCH_SIZE - pw_orig
    lr_padded = np.pad(lr_protected, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    patches = []
    for y in range(0, lr_padded.shape[0] - PATCH_SIZE + 1, stride):
        for x in range(0, lr_padded.shape[1] - PATCH_SIZE + 1, stride):
            patches.append(lr_padded[y:y + PATCH_SIZE, x:x + PATCH_SIZE, :])

    sr_patches = []
    for i in range(0, len(patches), BATCH_SIZE):
        batch = np.stack(patches[i:i + BATCH_SIZE])
        sr_patches.extend(list(model_sr.predict(batch, verbose=0)))

        prog = min(i + BATCH_SIZE, len(patches)) / len(patches)
        await context.bot.edit_message_text(f"SR Processing: {int(prog * 100)}%", chat_id, message_id)

    #  Hann Window Blending
    hr_h, hr_w = lr_padded.shape[0] * 2, lr_padded.shape[1] * 2
    final_hr = np.zeros((hr_h, hr_w, 3))
    weight = np.zeros((hr_h, hr_w, 3))

    win = np.outer(np.hanning(PATCH_SIZE * 2), np.hanning(PATCH_SIZE * 2))[..., None]

    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            y, x = iy * stride * 2, ix * stride * 2
            patch_hr = sr_patches[idx]
            final_hr[y:y + PATCH_SIZE * 2, x:x + PATCH_SIZE * 2] += patch_hr * win
            weight[y:y + PATCH_SIZE * 2, x:x + PATCH_SIZE * 2] += win
            idx += 1

    res = final_hr / (weight + 1e-8)

    offset = safety * 2
    final_h_real, final_w_real = h * 2, w * 2

    res_cropped = res[offset: offset + final_h_real, offset: offset + final_w_real]

    return Image.fromarray(((res_cropped + 1.0) * 127.5).clip(0, 255).astype(np.uint8))


# 5. TELEGRAM BOT


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Ciao! Send me an Image (as a Photo or a File) to restore it."
                                    "\n\n I can:\n - Dehaze using the Dark Channel Prior (DCP) technique.\n - Super-Resolution & Restoration using GAN (SRRGAN)."
                                    "\n\n You can also send me images as document to do not compress the quality!")


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if msg.document and msg.document.mime_type.startswith('image/'):
        file = await msg.document.get_file()
    elif msg.photo:
        file = await msg.photo[-1].get_file()
    else:
        return

    context.user_data['img_bytes'] = await file.download_as_bytearray()
    keyboard = [
        [InlineKeyboardButton("1. Dehaze Only", callback_data='dehaze')],
        [InlineKeyboardButton("2. SRRGAN Only", callback_data='sr')],
        [InlineKeyboardButton("3. Dehaze + SRRGAN", callback_data='combined')]
    ]
    await msg.reply_text("Choose Procedure:", reply_markup=InlineKeyboardMarkup(keyboard))


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Please send an image to begin!")


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id

    if 'img_bytes' not in context.user_data:
        await query.edit_message_text("Session expired. Please re-send the photo.")
        return

    original_pil = Image.open(io.BytesIO(context.user_data['img_bytes'])).convert("RGB")
    status = await context.bot.send_message(chat_id, "Processing.")

    try:
        if query.data == 'dehaze':
            await status.edit_text("Dehazing.")
            res_np = run_dcp_pipeline(np.array(original_pil))
            final_img = Image.fromarray(res_np)
            cap = "Dehazed (DCP)"
        elif query.data == 'sr':
            await status.edit_text("Restoring.")
            final_img = await run_sr_tiled(original_pil, context, chat_id, status.message_id)
            cap = "SRRGAN"
        elif query.data == 'combined':
            await status.edit_text("Step 1/2: Dehazing.")
            dehazed_pil = Image.fromarray(run_dcp_pipeline(np.array(original_pil)))
            await status.edit_text("Step 2/2: Restoring.")
            final_img = await run_sr_tiled(dehazed_pil, context, chat_id, status.message_id)
            cap = "Together: Dehazed + SRRGAN"

        bio = io.BytesIO()
        final_img.save(bio, format='PNG')
        bio.seek(0)

        await context.bot.send_document(chat_id, document=bio, filename="restored.png", caption=cap)
        await context.bot.delete_message(chat_id, status.message_id)
    except Exception as e:
        await status.edit_text(f"Error: {e}")


# 6. APPLICATION


if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image))
    app.add_handler(MessageHandler(filters.ALL & ~(filters.PHOTO | filters.Document.IMAGE), handle_text))
    app.add_handler(CallbackQueryHandler(button_callback))
    print("Bot is active.")
    app.run_polling()
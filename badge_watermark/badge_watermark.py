#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image, ImageDraw

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

import sys


badge_size = [500, 500]
margin = 100

def trim_image(input_image, new_size):
    old_size = input_image.size
    x0 = (old_size[0] - new_size[0]) / 2
    y0 = (old_size[1] - new_size[1]) / 2

    crop_box = (x0, y0, x0 + new_size[0], y0 + new_size[1])

    return input_image.crop(crop_box)


def mask_round_image(input_image, margin):
    supersampling_size = (input_image.size[0] * 4, input_image.size[1] * 4)
    mask = Image.new('L', supersampling_size, 0)
    draw = ImageDraw.Draw(mask)
    box = [margin, margin, supersampling_size[0] - margin, supersampling_size[1] - margin]
    draw.ellipse(box, fill=255)
    mask = mask.resize(input_image.size, Image.ANTIALIAS)

    masked_image = Image.new('RGBA', input_image.size)
    masked_image.paste(input_image, mask = mask)

    return masked_image


def watermark_image(input_image, watermark_image):
    watermarked_image = Image.alpha_composite(input_image, watermark_image)
    return watermarked_image


def main(argv):

    watermark_filepath = argv[1]
    watermark = Image.open(watermark_filepath)
    watermark_alpha = Image.new("RGBA", watermark.size)
    watermark = Image.blend(watermark, watermark_alpha, 0)

    for input_image_file in argv[2:]:
        input_image = Image.open(input_image_file)
        image = input_image.convert('RGBA')
        image = watermark_image(image, watermark)
        image = trim_image(image, badge_size)
        image = mask_round_image(image, margin)

        file_components = input_image_file.rsplit( ".", 1 )
        output_file = "{}_watermark.png".format(file_components[0])
        image.save(output_file)

if __name__ == "__main__":
    main(sys.argv)


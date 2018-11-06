#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image, ImageDraw

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

import sys


image_size = 51 * mm
image_locations = {'x':23, 'y':10, 'columns':3, 'rows':5}
image_margin = 5 * mm
scale = 0.9167


def generate_pdf(image, image_size, image_locations, image_margin, output_file):
    c = canvas.Canvas(output_file, pagesize=A4)
    width, height = A4

    x0 = image_locations['x'] * mm
    y0 = height - image_size - image_locations['y'] * mm

    image_reader = ImageReader(image)
    image_delta = image_size + image_margin

    for column in range(0, image_locations['columns']):
        for row in range(0, image_locations['rows']):
            x = x0 + image_delta * column
            y = y0 - image_delta * row
            c.drawImage(image_reader, x, y, width = image_size, height = image_size, mask='auto')

    c.showPage()
    c.save()


def scale_image(input_image, scale):
    scaled_size = (int(input_image.size[0] * (1/scale)), int(input_image.size[1] * (1/scale)))
    scale_offset_x = int((scaled_size[0] - input_image.size[0]) / 2)
    scale_offset_y = int((scaled_size[1] - input_image.size[1]) / 2)
    scaled_rect = (scale_offset_x, scale_offset_y)

    background_rgb = input_image.getpixel((1, 1))
    scaled_image = Image.new('RGBA', scaled_size, background_rgb)
    scaled_image.paste(input_image, box = scaled_rect)

    return scaled_image


def trim_image(input_image):
    supersampling_size = (input_image.size[0] * 4, input_image.size[1] * 4)
    mask = Image.new('L', supersampling_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + supersampling_size, fill=255)
    mask = mask.resize(input_image.size, Image.ANTIALIAS)

    trimed_image = Image.new('RGBA', input_image.size)
    trimed_image.paste(input_image, mask = mask)

    return trimed_image


def main(argv):
    for input_image_file in argv[1:]:
        output_file="{}.pdf".format(input_image_file.rsplit( ".", 1 )[ 0 ])

        input_image = Image.open(input_image_file)
        if scale is not 1:
            input_image = scale_image(input_image, scale)
        input_image = trim_image(input_image)

        generate_pdf(input_image, image_size, image_locations, image_margin, output_file)


if __name__ == "__main__":
    main(sys.argv)


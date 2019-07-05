#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from copy import copy
import functools
import math
import json
import operator
import os

from PIL import Image, ImageDraw, ImageFont
# Workaraund for a bug in reportlab
Image.VERSION = '6.0.0'

from reportlab.lib.pagesizes import A4, A3
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

import sys

def hex_to_rgb(hex_str):
    h = hex_str.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2 ,4))

class Bundle:

    RESERVED_COUNTERS = [
        "batch.index",
        "bundle.index",
    ]

    @staticmethod
    def parse_from_json(config_path, debug = False):

        with open(config_path) as config_file:
            basedir = os.path.dirname(config_file.name)
            json_config = json.load(config_file)

        page = Page.parse_from_json(basedir, json_config)

        global_vars = json_config['vars']

        batches = Batch.parse_from_json(json_config)

        templates_path = os.path.join(basedir, page.templates)
        with open(templates_path) as templates_file:
            json_templates = json.load(templates_file)
        templates = Template.parse_from_json(json_templates)

        return Bundle(basedir, page, global_vars, batches, templates, debug)


    def __init__(self, basedir, page, global_vars, batches, templates, debug = False):
        self.basedir = basedir
        self.page = page
        self.global_vars = global_vars
        self.batches = batches
        self.templates = templates
        self.counters = {}
        self.debug = debug

        for batch in self.batches:
            batch.bundle = self
            batch.comp_vars = batch.compute_vars()

        for template in self.templates.values():
            template.bundle = self

            for layer in template.layers:
                layer.bundle = self


    def increment_counter(self, counter_name):
        if counter_name not in self.counters.keys():
            self.counters[counter_name] = 0

        if counter_name not in Bundle.RESERVED_COUNTERS:
            self.counters[counter_name] += 1

        return self.counters[counter_name]


    def generate_pdf(self, output_folder, page_format, orientation):
        non_empty_batches = {}
        self.counters = {}
        for batch in self.batches:
            if batch.count > 0:
                batch_set_id = batch.set
                if batch_set_id not in non_empty_batches.keys():
                    non_empty_batches[batch_set_id] = []
                batch_set = non_empty_batches[batch_set_id]
                batch_set.append(batch)

        for batch_set in non_empty_batches.values():
            card_size = self.templates[batch_set[0].templates[0]].size
            self.generate_pdf_for(batch_set, output_folder, card_size, page_format, orientation)


    def generate_pdf_for(self, batches, output_folder, card_size, page_format, orientation):
        output_file = "{}.pdf".format(batches[0].set)
        output_path_pdf = os.path.join(output_folder, "pdf")
        output_path_file = os.path.join(output_path_pdf, output_file)

        if not os.path.exists(output_path_pdf):
            os.makedirs(output_path_pdf)

        print("     Working on {}".format(output_file))

        page_size = A4
        if page_format == 'A3':
            page_size = A3
        if orientation == 'landscape':
            page_size = (page_size[1], page_size[0])
        c = canvas.Canvas(output_path_file, pagesize=page_size)
        width, height = page_size

        available_width = width - (self.page.margin[0] * mm)
        available_height = height - (self.page.margin[1] * mm)

        column_width = card_size[0] * mm
        row_height = card_size[1] * mm

        columns = int (available_width / column_width)
        rows = int (available_height / row_height)
        cards_per_page = columns * rows

        card_count = 0
        batch_counts = set()
        for batch in batches:
            card_count += batch.count * len(batch.templates)
            batch_counts.add(len(batch.templates))
        page_count = math.ceil(card_count / cards_per_page)

        # page_count must be multiple of the number templates in batches.
        multiplier = functools.reduce(operator.mul, list(batch_counts), 1)
        page_count = math.ceil(page_count / multiplier) * multiplier

        # This leave empty spaces on the last cards.
        # Fill the gaps with the latest batch.
        last_batch = batches[-1]
        empty_spaces = (page_count * cards_per_page) - card_count
        extra_cards_count = empty_spaces / len(last_batch.templates)
        last_batch.count += extra_cards_count

        # Generate base images for cards
        image_readers = {}
        for batch in batches:
            for template_name in batch.templates:
                template = self.templates[template_name]
                image = self.generate_image_for(batch, template)
                if not batch.id in image_readers.keys():
                    image_readers[batch.id] = {}
                image_readers[batch.id][template_name] = ImageReader(image)

        # Order cards on pages
        pages = [None] * page_count
        page = 0
        column = 0
        row = 0
        initial_card = 0
        cards_processed = 0
        bundle_index = 1
        for batch in batches:
            batch_index = 1
            template_count = len(batch.templates)
            initial_card = cards_processed

            while (cards_processed - initial_card) < (batch.count * template_count):
                for template_name in batch.templates:
                    # Initialize counter
                    # Initialize page
                    if row == 0 and column == 0:
                        pages[page] = [None] * rows
                        for i in range(0, len(pages[page])):
                            pages[page][i] = [None] * columns
                    # Odd pages and Even should be mirrored
                    row_column = column
                    if page % 2 == 1:
                        row_column = columns - column - 1
                    # Set the propper template
                    card = {
                        "batch": batch,
                        "template": template_name,
                        "counters": {
                            "batch.index": batch_index,
                            "bundle.index": bundle_index,
                        }
                    }
                    pages[page][row][row_column] = card
                    last_card = card
                    # Imcrement counters
                    page += 1
                    if page >= page_count:
                        page = 0
                        column += 1
                    if column >= columns:
                        column = 0
                        row += 1
                    cards_processed +=1

                # Increase the card index inside a bundle, AFTER all the templates for that card are rendered.
                batch_index +=1
                bundle_index +=1

        # Graphic positions
        mul = 10
        c0 = columns * mul // 2 * -1
        cn = (c0 * -1)

        r0 = rows * mul // 2 * -1
        rn = (r0 * -1)

        margin_x = math.floor(float(available_width - column_width * columns) / columns)
        margin_y = math.floor(float(available_height - row_height * rows) / columns)

        # Paint cards
        page = 0
        while page < page_count:
            print("Starting Page {} of {}".format(page + 1, page_count))
            row = 0
            for row_x in range(rn-mul, r0-mul, -mul):
                column = 0
                for column_y in range(c0, cn, mul):
                    card = pages[page][row][column]
                    if card:
                        batch = card['batch']
                        template_name = card['template']
                        template = self.templates[template_name]

                        # Delta y to adjust printer fails
                        delta_y = 0
                        delta_x = 0
                        for offset_key in self.page.templates_offsets.keys():
                            if offset_key in template.name:
                                offset = self.page.templates_offsets[offset_key]
                                delta_x = offset[0] * mm
                                delta_y = offset[1] * mm

                        x = (width / 2) + (float(column_y) / mul * (column_width + margin_x)) + margin_x / 2
                        y = (height / 2) + (float(row_x) / mul * (row_height + margin_y)) + margin_y / 2
                        x += delta_x
                        y += delta_y
                        x = int(x)
                        y = int(y)

                        # Reset global counters
                        for counter_name, counter_value in card['counters'].items():
                            self.counters[counter_name] = counter_value

                        # Bleeding
                        for bleeding_layer in template.bleeding:
                            bleeding_layer.render_over(x, y, c, template)

                        # Re-Generate image with counters
                        if batch.raffle:
                            image = self.generate_image_for(batch, template)
                            image_readers[batch.id][template.name] = ImageReader(image)

                        # Image
                        image_reader = image_readers[batch.id][template.name]
                        c.drawImage(image_reader, x, y, width = column_width, height = row_height, mask='auto')

                        # Debug
                        if self.debug:
                            c.drawString(x, y, "{}_{}".format(self.counters["bundle.index"], self.counters["batch.index"]))

                    column += 1
                row += 1
            c.showPage()
            page += 1

        c.save()

    def generate_images(self, output_folder):
        self.counters = {}
        for batch in self.batches:
            if batch.count <= 0:
                continue
            for template_name in batch.templates:
                template = self.templates[template_name]
                self.generate_image_for(batch, template, True, output_folder)


    def generate_image_for(self, batch, template, save = False, output_folder = ''):
        image = template.image_for(batch)

        if save:
            output_path_img = os.path.join(output_folder, "img")
            if not os.path.exists(output_path_img):
                os.makedirs(output_path_img)

            image_name = "{}_{}_{}.png".format(batch.id, batch.set, template.name)
            image_path = os.path.join(output_path_img, image_name)
            image.save(image_path)

        return image

class Page:

    @staticmethod
    def parse_from_json(basedir, json_config):
        return Page(
            json_config['page']['margin'],
            json_config['page']['units'],
            json_config['page']['templates'],
            json_config['page']['templates_offsets'])


    def __init__(self, margin, units, templates, templates_offsets):
        self.margin = margin
        self.units = units
        self.templates = templates
        self.templates_offsets = templates_offsets


class Batch:

    @staticmethod
    def parse_from_json(json_config):
        batches = []
        for batch_config in json_config['batches']:
            batch_set = batch_config['set']
            batch_id = batch_config['id']
            count = batch_config['count']
            raffle = batch_config['raffle'] if 'raffle' in batch_config.keys() else False
            mosaic_columns = batch_config['mosaic_columns'] if 'mosaic_columns' in batch_config.keys() else 1
            local_vars = batch_config['vars']
            templates = batch_config['templates']
            batches.append(Batch(batch_set, batch_id, count, raffle, mosaic_columns, local_vars, templates))
        return batches


    def __init__(self,batch_set, batch_id, count, raffle, mosaic_columns, local_vars, templates):
        self.bundle = None
        self.set = batch_set
        self.id = batch_id
        self.count = count
        self.raffle = raffle
        self.mosaic_columns = mosaic_columns
        self.local_vars = local_vars
        self.templates = templates

        self.comp_vars = {}


    def compute_vars(self):
        comp_vars = {}
        comp_vars.update(self.bundle.global_vars.copy())
        comp_vars.update(self.local_vars.copy())
        return comp_vars

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result


class Template:

    @staticmethod
    def parse_from_json(json_config):
        templates = {}
        for template in json_config['templates']:
            name = template['name']
            background = template['background']
            size = template['size']
            layers = Layer.parse_from_json(template['layers'])
            bleeding = PDFLayer.parse_from_json(template['bleeding'])
            templates[name] = Template(name, background, size, layers, bleeding)
        return templates


    def __init__(self, name, background, size, layers, bleeding):
        self.bundle = None
        self.name = name
        self.background = background
        self.size = size
        self.layers = layers
        self.bleeding = bleeding


    def image_for(self, batch):
        image_path = os.path.join(self.bundle.basedir, self.background)
        image = Image.open(image_path)

        res = float(image.size[0]) / self.size[0]

        for layer in self.layers:
            image = layer.render_over(image, batch, res)

        return image


class Layer:

    @staticmethod
    def parse_from_json(json_layers):
        layers = []
        for json_layer in json_layers:
            layer_type = json_layer['type']
            if layer_type == 'text':
                layers.append(TextLayer.parse_from_json(json_layer))
            elif layer_type == 'text_counter':
                layers.append(TextCounterLayer.parse_from_json(json_layer))
            elif layer_type == 'image':
                layers.append(ImageLayer.parse_from_json(json_layer))
            elif layer_type == 'image_mosaic':
                layers.append(ImageMosaicLayer.parse_from_json(json_layer))
        return layers

    def render_over(self, image, batch, res):
        raise NotImplementedError


class TextLayer(Layer):

    @staticmethod
    def parse_from_json(json_layer):
        layer_type = json_layer['type']
        if layer_type != 'text':
            raise NotImplementedError

        rotation = json_layer['rotation'] if 'rotation' in json_layer.keys() else 0


        return TextLayer(
            json_layer['text'],
            json_layer['font_name'],
            json_layer['font_size'],
            json_layer['font_color'],
            json_layer['text_x'],
            json_layer['text_y'],
            json_layer['text_align'],
            rotation)


    def __init__(self, text, font_name, font_size, font_color, text_x, text_y, text_align, rotation):
        self.bundle = None
        self.text = text
        self.font_name = font_name
        self.font_size = font_size
        self.font_color = font_color
        self.text_x = text_x
        self.text_y = text_y
        self.text_align = text_align
        self.rotation = rotation


    def render_over(self, image, batch, res):
        texts_vals = []
        for text in self.text:
            texts_vals.append(batch.comp_vars[text])
        text = ''.join(texts_vals)

        if self.rotation:
            return TextLayer.render_rotated_text_over(image, batch, res, text, self.font_name, self.font_size, self.font_color, self.text_x, self.text_y, self.text_align, self.rotation)
        else:
            return TextLayer.render_text_over(image, batch, res, text, self.font_name, self.font_size, self.font_color, self.text_x, self.text_y, self.text_align)

    @staticmethod
    def render_rotated_text_over(image, batch, res, text, font_name, font_size, font_color, text_x, text_y, text_align, rotation):
        # Prepare
        font = ImageFont.truetype(font_name, int(font_size))
        spacing = 1
        draw = ImageDraw.Draw(image)
        text_size = draw.textsize(text, font=font, spacing=spacing)

        # Image for text to be rotated
        img_txt = Image.new('RGBA', text_size, (255, 255, 255, 0))
        draw_txt = ImageDraw.Draw(img_txt)

        draw_txt.text([0,0], text, font=font, fill=hex_to_rgb(font_color), spacing=spacing, align='right')
        rotated_text_img = img_txt.rotate(rotation, expand=1)

        rotated_text_width, rotated_text_height = rotated_text_img.size
        orig = [int(text_x * res), int(text_y * res)]
        if text_align == 'right': # Right Top
            orig[0] -= rotated_text_width
        elif text_align == 'left': # Left Bottom
            orig[1] -= rotated_text_height
        elif text_align == 'center': # Center X / Y
            orig[0] -= int(rotated_text_width / 2)
            orig[1] -= int(rotated_text_height / 2)

        image.paste(rotated_text_img, orig, rotated_text_img)

        return image

    @staticmethod
    def render_text_over(image, batch, res, text, font_name, font_size, font_color, text_x, text_y, text_align):
        # Prepare
        font = ImageFont.truetype(font_name, int(font_size))

        #Draw
        draw = ImageDraw.Draw(image)
        orig = [int(text_x * res), int(text_y * res)]
        spacing = 1

        text_size = draw.textsize(text, font=font, spacing=spacing)
        orig[1] -= text_size[1]
        if text_align == 'right':
            orig[0] -= text_size[0]
        elif text_align == 'center':
            orig[0] -= int(text_size[0] / 2)

        draw.text(orig, text, font=font, fill=hex_to_rgb(font_color), spacing=spacing, align=text_align)

        return image



class TextCounterLayer(Layer):

    @staticmethod
    def parse_from_json(json_layer):
        layer_type = json_layer['type']
        if layer_type != 'text_counter':
            raise NotImplementedError

        rotation = json_layer['rotation'] if 'rotation' in json_layer.keys() else 0

        return TextCounterLayer(
            json_layer['counter'],
            json_layer['alt_text'],
            json_layer['font_name'],
            json_layer['font_size'],
            json_layer['alt_font_size'],
            json_layer['font_color'],
            json_layer['text_x'],
            json_layer['text_y'],
            json_layer['text_align'],
            rotation)


    def __init__(self, counter, alt_text, font_name, font_size, alt_font_size, font_color, text_x, text_y, text_align, rotation):
        self.bundle = None
        self.counter = counter
        self.alt_text = alt_text
        self.font_name = font_name
        self.font_size = font_size
        self.alt_font_size = alt_font_size
        self.font_color = font_color
        self.text_x = text_x
        self.text_y = text_y
        self.text_align = text_align
        self.rotation = rotation


    def render_over(self, image, batch, res):
        text = batch.comp_vars[self.alt_text]
        font_size = self.alt_font_size
        if batch.raffle:
            if self.counter in Bundle.RESERVED_COUNTERS:
                text = str(self.bundle.increment_counter(self.counter))
            else:
                text = str(self.bundle.increment_counter('{}_{}'.format(batch.id, self.counter)))

            font_size = self.font_size

        if self.rotation:
            return TextLayer.render_rotated_text_over(image, batch, res, text, self.font_name, font_size, self.font_color, self.text_x, self.text_y, self.text_align, self.rotation)
        else:
            return TextLayer.render_text_over(image, batch, res, text, self.font_name, font_size, self.font_color, self.text_x, self.text_y, self.text_align)


class ImageLayer(Layer):

    @staticmethod
    def parse_from_json(json_layer):
        layer_type = json_layer['type']
        if layer_type != 'image':
            raise NotImplementedError

        return ImageLayer(json_layer['src'], json_layer['img_x'], json_layer['img_y'], json_layer['width'], json_layer['height'])


    def __init__(self, src, img_x, img_y, width, height):
        self.bundle = None
        self.src = src
        self.img_x = img_x
        self.img_y = img_y
        self.width = width
        self.height = height


    def render_over(self, image, batch, res):
        img_src = batch.comp_vars[self.src]
        ImageLayer.render_image_over(image, batch, res, self.bundle.basedir, img_src, self.width, self.height, self.img_x, self.img_y)
        return image

    @staticmethod
    def render_image_over(image, batch, res, basedir, img_src, width, height, img_x, img_y):
        # Prepare origin image
        layer_image_path = os.path.join(basedir, img_src)
        layer_image = Image.open(layer_image_path)

        # Resize to fit
        px_width = width * res
        px_height = height * res
        if layer_image.size[0] < width * res or layer_image.size[1] < height * res:
            scale_x = layer_image.size[0] / px_width
            scale_y = layer_image.size[1] / px_height
            scale = scale_x if scale_x < scale_y else scale_y
            layer_image = layer_image.resize((int(layer_image.size[0] / scale) , int(layer_image.size[1] / scale)))
        layer_image.thumbnail([int(width * res), int(height * res)])

        # New image coordinates
        x_delta = (width * res - layer_image.size[0]) / 2
        y_delta = (height * res - layer_image.size[1]) / 2
        x = (img_x * res) + x_delta
        y = (img_y * res) + y_delta
        box = (int(x), int(y))

        # Paint into target image
        image.paste(layer_image, box=box)


class ImageMosaicLayer(Layer):

    @staticmethod
    def parse_from_json(json_layer):
        layer_type = json_layer['type']
        if layer_type != 'image_mosaic':
            raise NotImplementedError

        return ImageMosaicLayer(json_layer['src'], json_layer['img_x'], json_layer['img_y'], json_layer['width'], json_layer['height'])


    def __init__(self, src, img_x, img_y, width, height):
        self.bundle = None
        self.src = src
        self.img_x = img_x
        self.img_y = img_y
        self.width = width
        self.height = height


    def render_over(self, image, batch, res):
        img_sources = batch.comp_vars[self.src]
        if len(img_sources) <= 0:
            return image

        columns = batch.mosaic_columns
        rows = math.ceil(float(len(img_sources))/ columns)

        cell_width = self.width / columns
        cell_height = self.height / rows

        row = 0
        column = 0
        for image_src in img_sources:
            x = self.img_x + column * cell_width
            y = self.img_y + row * cell_height
            ImageLayer.render_image_over(image, batch, res, self.bundle.basedir, image_src, cell_width, cell_height, x, y)

            column += 1
            if column >= columns:
                column = 0
                row += 1

        return image


class PDFLayer:

    @staticmethod
    def parse_from_json(json_bleeding_layers):
        layers = []
        for json_layer in json_bleeding_layers:
            layer_type = json_layer['type']
            if layer_type == 'fill_color':
                layers.append(FillColorLayer.parse_from_json(json_layer))
            elif layer_type == 'cut_lines':
                layers.append(CutLinesLayer.parse_from_json(json_layer))
        return layers

    def render_over(self, canvas, template):
        raise NotImplementedError


class FillColorLayer(Layer):

    @staticmethod
    def parse_from_json(json_layer):
        layer_type = json_layer['type']
        if layer_type != 'fill_color':
            raise NotImplementedError

        return FillColorLayer(json_layer['color'], json_layer['x'], json_layer['y'], json_layer['width'], json_layer['height'])


    def __init__(self, color, x, y, width, height):
        self.color = color
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def render_over(self, x, y, canvas, template):
        color = hex_to_rgb(self.color)
        canvas.setFillColorRGB(color[0]/255.0, color[1]/255.0, color[2]/255.0)
        canvas.rect(x + (self.x * mm), y + (self.y * mm), self.width * mm, self.height * mm, fill=1)

        canvas.setFillColorRGB(0,0,0)


class CutLinesLayer(Layer):

    @staticmethod
    def parse_from_json(json_layer):
        layer_type = json_layer['type']
        if layer_type != 'cut_lines':
            raise NotImplementedError

        colors = json_layer['colors'] if 'colors' in json_layer.keys() else ['000000', '000000', '000000', '000000']

        return CutLinesLayer(json_layer['margins'][0], json_layer['margins'][1], colors)


    def __init__(self, margin0, margin1, colors):
        self.margin0 = margin0
        self.margin1 = margin1
        self.colors = colors

    def render_over(self, x, y, canvas, template):
        column_width = template.size[0] * mm
        row_height = template.size[1] * mm

        color =  hex_to_rgb(self.colors[3])
        canvas.setStrokeColorRGB(color[0]/255.0, color[1]/255.0, color[2]/255.0)
        canvas.line(x - (self.margin0 * mm), y, x - (self.margin1 * mm), y)
        canvas.line(x, y - (self.margin0 * mm), x , y - (self.margin1 * mm))

        color =  hex_to_rgb(self.colors[0])
        canvas.setStrokeColorRGB(color[0]/255.0, color[1]/255.0, color[2]/255.0)
        canvas.line(x - (self.margin0 * mm), y + row_height + (self.margin1 * mm), x - (self.margin1 * mm), y + row_height + (self.margin1 * mm))
        canvas.line(x, y + row_height + (self.margin0 * mm), x, y + row_height + (self.margin1 * mm))

        color =  hex_to_rgb(self.colors[2])
        canvas.setStrokeColorRGB(color[0]/255.0, color[1]/255.0, color[2]/255.0)
        canvas.line(x + column_width + (self.margin1 * mm), y, x + column_width + (self.margin0 * mm), y)
        canvas.line(x + column_width, y - (self.margin0 * mm), x + column_width , y - (self.margin1 * mm))

        color =  hex_to_rgb(self.colors[1])
        canvas.setStrokeColorRGB(color[0]/255.0, color[1]/255.0, color[2]/255.0)
        canvas.line(x + column_width + (self.margin1 * mm), y + row_height + (self.margin1 * mm), x + column_width + (self.margin0 * mm), y + row_height + (self.margin1 * mm))
        canvas.line(x + column_width, y + row_height + (self.margin0 * mm), x + column_width, y + row_height + (self.margin1 * mm))


def main(argv):
    parser = argparse.ArgumentParser(description='Generate cards from config file.')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Dump debug information in the pdfs')
    parser.add_argument('-c', '--configs', nargs='+', required=True,
                        help='Config files to process')
    parser.add_argument('-p', '--generate-pdf', action='store_true',
                        help='Generate a ready to print pdf')
    parser.add_argument('-i', '--generate-png', action='store_true',
                        help='Generate one png image for every type of card')
    parser.add_argument('-s', '--pdf-size', choices=['A4', 'A3'], default='A3',
                        help='If output is a pdf file, the size of the pdf pages')
    parser.add_argument('-r', '--pdf-orientation', choices=['portrait', 'landscape'], default='landscape',
                        help='If output is a pdf file, the orientation of the pdf pages')
    parser.add_argument('-o', '--output-folder', default=os.getcwd(),
                        help='Output folder where to store the cards. Defaults to the current directory.')

    args = parser.parse_args()

    for config_path in args.configs:
        print("Processing file: {}".format(config_path))
        bundle = Bundle.parse_from_json(config_path, debug = args.debug)
        if args.generate_png:
            bundle.generate_images(output_folder=args.output_folder)
        if args.generate_pdf:
            bundle.generate_pdf(
                output_folder=args.output_folder,
                page_format=args.pdf_size,
                orientation=args.pdf_orientation,
            )

if __name__ == "__main__":
    main(sys.argv)


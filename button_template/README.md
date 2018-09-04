# Button Template Maker

This tool creates A4 PDFs ready to print and use to create buttons / badges.

It currently only supports 38mm buttons.

You should be able to use the resulting files on chapea.com.

This is only on draft status, so there are LOTS of things to improve.

# Install

I've tested this with python 2.7, Pillow 5.2.0 and reportlab 3.5.6.

```
pip install Pillow
pip install reportlab
```

Depending on your system you might need libjpeg to work on .jpg files and Pillow.

# Usage

Just pass as an argument the files you want to use, and the tool will create a .pdf file next to the source file:

```
python create_template.py source_image.png
```

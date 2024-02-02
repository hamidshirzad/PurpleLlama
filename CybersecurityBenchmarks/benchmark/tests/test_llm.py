# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest

from .. import llm


class TestLLM(unittest.TestCase):
    def test_create(self) -> None:
        llama = llm.create("TOGETHER::togethercomputer/llama-2-7b::<my API key>")
        self.assertTrue(isinstance(llama, llm.TOGETHER))
        self.assertEqual(llama.model, "togethercomputer/llama-2-7b")
        self.assertEqual(llama.api_key, "<my API key>")

        self.assertRaises(ValueError, lambda: llm.create("GPT::unknown::<my API key"))
from PIL import Image, ImageDraw, ImageFont

# Create an image with white background
img = Image.new('RGB', (500, 500), color = 'white')
d = ImageDraw.Draw(img)

# Define colors and shapes for the llama
llama_color = 'beige'
heart_color = 'red'

# Draw the body of the llama
d.ellipse((150, 250, 350, 450), fill=llama_color)

# Draw the head of the llama
d.ellipse((200, 150, 300, 250), fill=llama_color)

# Draw the legs of the llama
for i in range(170, 351, 60):
    d.rectangle((i, 350, i + 20, 450), fill=llama_color)

# Draw the ears of the llama
d.polygon([(225, 150), (235, 100), (245, 150)], fill=llama_color)
d.polygon([(255, 150), (265, 100), (275, 150)], fill=llama_color)

# Draw eyes of the llama
d.ellipse((230, 190, 240, 200), fill='black')
d.ellipse((260, 190, 270, 200), fill='black')

# Draw a big heart for the love theme
d.heart((200, 200, 300, 300), fill=heart_color)

# Save the image to a file
img.save('love_llama.png')

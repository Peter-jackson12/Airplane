"""Render original SVG artwork into a compact looping GIF (optional Pillow/CairoSVG).

Run with Python plus Pillow and CairoSVG installed. No network or model fitting.
"""
from pathlib import Path
from io import BytesIO
import cairosvg
from PIL import Image
from cat_playground import play_frame
OUT=Path(__file__).resolve().parents[1]/'assets/readme'

def build():
    frames=[]
    for i in range(60):
        svg=play_frame(i/60)
        png=cairosvg.svg2png(bytestring=svg.encode(),output_width=1000,output_height=280)
        rgba=Image.open(BytesIO(png)).convert('RGBA')
        frame=Image.new('RGB',rgba.size,'#FFF9EF')
        frame.paste(rgba,mask=rgba.getchannel('A'))
        frames.append(frame)
    frames[0].save(OUT/'cat_playground.gif',save_all=True,append_images=frames[1:],duration=70,loop=0,optimize=True,disposal=2)
    (OUT/'cat_playground_still.svg').write_text(play_frame(0))
    print('60 frames / 4.2 seconds / infinite loop; original SVG artwork.')
if __name__=='__main__':build()

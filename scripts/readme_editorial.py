"""Shared evidence-first F Korean Air homepage-inspired SVG primitives; stdlib only and no external assets."""
from html import escape

# Observed 2026-10-01 at https://www.koreanair.com/: navy, cyan, pale sky, white.
# ACCENT is a darker cyan adaptation for small labels; not an official brand token.
PAPER='#FFFFFF'; INK='#051766'; MUTED='#5E5E5E'; RULE='#D9D9D9'; ACCENT='#004766'
CYAN='#57BBEB'; SKY='#DDF1FB'; HIGHLIGHT='#F7F7F7'
SANS='Noto Sans CJK KR, Malgun Gothic, Apple SD Gothic Neo, sans-serif'
HEADING=SANS
# F evidence hierarchy: 1600-unit canvas, 80-unit inset, 8-unit spacing rhythm.
# At 896px README width, a 28-unit annotation is 15.7px.
TYPE = {'caption': 28, 'body': 32, 'section': 44, 'title': 64, 'result': 112}
SPACE = {'unit': 8, 'inset': 80, 'row': 96}


class SVG:
    def __init__(self,h,title,desc,header_band=True):
        self.h=h
        self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="{h}" viewBox="0 0 1600 {h}" role="img" aria-labelledby="title desc">',f'<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>',f'<rect width="1600" height="{h}" fill="{PAPER}"/>']
        if header_band:
            self.rect(0,0,1600,88,SKY)
    def rect(self,x,y,w,h,color):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}"/>')
    def text(self,x,y,t,size=30,color=INK,weight=400,family=SANS,anchor='start',spacing=None):
        ls=f' letter-spacing="{spacing}"' if spacing is not None else ''
        self.parts.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="{family}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" style="font-variant-numeric:tabular-nums"{ls}>{escape(t)}</text>')
    def line(self,x1,y1,x2,y2,color=RULE,w=1):self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{w}"/>')
    def path(self,d,color=RULE,w=2):self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w}"/>')
    def circle(self,x,y,r,fill=PAPER,stroke=INK,w=2):self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{w}"/>')
    def running(self,left,right):
        self.text(84,60,left,23,weight=500)
        self.text(1516,60,right,19,color=MUTED,anchor='end',spacing=1.8)
        self.line(84,86,1516,86,CYAN,2)
    def save(self,path):
        path.write_text('\n'.join(self.parts+['</svg>'])+'\n',encoding='utf-8')


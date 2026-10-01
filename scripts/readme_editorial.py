"""Shared flat editorial SVG primitives; stdlib only and no external assets."""
from html import escape

PAPER='#F5F2EB'; INK='#20201E'; MUTED='#62635D'; RULE='#D4D1C8'; RUST='#A5442B'
SANS='Noto Sans CJK KR, Malgun Gothic, Apple SD Gothic Neo, sans-serif'
SERIF='Noto Serif CJK KR, Nanum Myeongjo, Batang, serif'

class SVG:
    def __init__(self,h,title,desc):
        self.h=h
        self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="{h}" viewBox="0 0 1600 {h}" role="img" aria-labelledby="title desc">',f'<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>',f'<rect width="1600" height="{h}" fill="{PAPER}"/>']
    def text(self,x,y,t,size=30,color=INK,weight=400,family=SANS,anchor='start',spacing=None):
        ls=f' letter-spacing="{spacing}"' if spacing is not None else ''
        self.parts.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="{family}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" style="font-variant-numeric:tabular-nums"{ls}>{escape(t)}</text>')
    def line(self,x1,y1,x2,y2,color=RULE,w=1):self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{w}"/>')
    def path(self,d,color=RULE,w=2):self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w}"/>')
    def circle(self,x,y,r,fill=PAPER,stroke=INK,w=2):self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{w}"/>')
    def running(self,left,right):
        self.text(84,60,left,23,weight=500)
        self.text(1516,60,right,19,color=MUTED,anchor='end',spacing=1.8)
        self.line(84,86,1516,86,INK,2)
    def save(self,path):
        path.write_text('\n'.join(self.parts+['</svg>'])+'\n',encoding='utf-8')


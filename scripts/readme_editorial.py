"""H technical-announcement primitives. Original local vectors, stdlib only."""
from html import escape
PAPER='#FFFFFF'; INK='#202124'; MUTED='#5F6368'; RULE='#DADCE0'; ACCENT='#185ABC'
CYAN='#D2E3FC'; SKY='#E8F0FE'; HIGHLIGHT='#F8F9FA'
SANS='Noto Sans CJK KR, Malgun Gothic, Apple SD Gothic Neo, sans-serif'
HEADING=SANS
TYPE={'caption':28,'body':32,'section':44,'title':64,'result':112}
SPACE={'unit':8,'inset':80,'row':96}
class SVG:
    def __init__(self,h,title,desc,header_band=False):
        self.h=h
        self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="{h}" viewBox="0 0 1600 {h}" role="img" aria-labelledby="title desc" data-design="technical-launch-h">',f'<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>',f'<rect width="1600" height="{h}" fill="{PAPER}"/>']
    def rect(self,x,y,w,h,color,rx=0):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}" rx="{rx}"/>')
    def text(self,x,y,t,size=30,color=INK,weight=400,family=SANS,anchor='start',spacing=None):
        ls=f' letter-spacing="{spacing}"' if spacing is not None else ''
        self.parts.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="{family}" font-size="{max(28,size)}" font-weight="{weight}" text-anchor="{anchor}" style="font-variant-numeric:tabular-nums"{ls}>{escape(str(t))}</text>')
    def line(self,x1,y1,x2,y2,color=RULE,w=1):self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{w}"/>')
    def path(self,d,color=RULE,w=2):self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w}"/>')
    def circle(self,x,y,r,fill=PAPER,stroke=INK,w=2):self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{w}"/>')
    def save(self,path):path.write_text('\n'.join(self.parts+['</svg>'])+'\n',encoding='utf-8')

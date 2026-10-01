"""Shared deliberately tacky G Bonobono-PPT parody SVG primitives; stdlib only and no external assets."""
from html import escape

# Parody palette: candy pink, acid green, purple, yellow; no brand affiliation.
PAPER='#FFFDED'; INK='#221044'; MUTED='#503849'; RULE='#CC8FDC'; ACCENT='#6B0059'
CYAN='#98FF67'; SKY='#FFE1F5'; HIGHLIGHT='#FFF57B'
SANS='Noto Sans CJK KR, Malgun Gothic, Apple SD Gothic Neo, sans-serif'
HEADING='Noto Serif CJK KR, serif'
# Evidence remains in a 1600-unit coordinate system inside a deliberately garish frame.
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
        path.write_text(parody_frame('\n'.join(self.parts+['</svg>'])+'\n', path.stem),encoding='utf-8')



# This is original vector fan art for the requested Bonobono PPT meme parody.
# No third-party image files, fonts, scripts, or network resources are embedded.
def bonobono(x, y, scale=1):
    return f'''<g data-parody-mascot="original-vector-fan-art" transform="translate({x} {y}) scale({scale})">
<ellipse cx="0" cy="70" rx="105" ry="117" fill="#70BFFF" stroke="#243242" stroke-width="4"/>
<ellipse cx="-57" cy="-41" rx="31" ry="37" fill="#70BFFF" stroke="#243242" stroke-width="4"/>
<ellipse cx="57" cy="-41" rx="31" ry="37" fill="#70BFFF" stroke="#243242" stroke-width="4"/>
<ellipse cx="0" cy="0" rx="118" ry="102" fill="#70BFFF" stroke="#243242" stroke-width="4"/>
<circle cx="-56" cy="-6" r="8" fill="#17202A"/><circle cx="56" cy="-6" r="8" fill="#17202A"/>
<ellipse cx="-28" cy="39" rx="37" ry="26" fill="#FFFDF3"/><ellipse cx="28" cy="39" rx="37" ry="26" fill="#FFFDF3"/>
<ellipse cx="0" cy="22" rx="17" ry="13" fill="#17202A"/>
<path d="M0 34 V58 M-71 35 l-35 -8 M-70 49 l-36 9 M71 35 l35 -8 M70 49 l36 9" fill="none" stroke="#17202A" stroke-width="3"/>
<ellipse cx="-68" cy="113" rx="33" ry="22" fill="#70BFFF" stroke="#243242" stroke-width="3"/>
<ellipse cx="68" cy="113" rx="33" ry="22" fill="#70BFFF" stroke="#243242" stroke-width="3"/>
<path d="M-48 118 Q-55 67 -34 72 Q-19 50 0 64 Q19 50 34 72 Q55 67 48 118 Z" fill="#FF9FCC" stroke="#6B0059" stroke-width="4"/>
<path d="M0 114 V70 M-18 112 l-13 -34 M18 112 l13 -34" fill="none" stroke="#B93188" stroke-width="3"/>
<path d="M113 -53 Q94 -18 115 -17 Q136 -19 113 -53" fill="#40DEFF" stroke="#17446D" stroke-width="3"/>
</g>'''


def wordart(text, x, y, size=72, angle=-3):
    colors=['#EF005D','#EE6A00','#816900','#008C33','#0066EE','#8200B5']
    letters=''.join(f'<tspan fill="{colors[i%len(colors)]}">{escape(c)}</tspan>' for i,c in enumerate(text))
    attrs=f'x="{x}" y="{y}" font-size="{size}" font-family="Noto Sans CJK KR, sans-serif" font-weight="900"'
    return f'<g data-parody-wordart="true" transform="rotate({angle} {x} {y})"><text {attrs} dx="7" dy="9" fill="#291142" stroke="#291142" stroke-width="10">{escape(text)}</text><text {attrs} stroke="#FFFCE2" stroke-width="4" paint-order="stroke">{letters}</text></g>'


def parody_frame(raw, name):
    """Decorative parody outside a translated, unscaled numerical chart.

    Plot positions and axis labels retain the F figure coordinate system.
    Loud gradients and mascots never encode measured values.
    """
    import re
    height=int(re.search(r'height="(\d+)"',raw).group(1))
    header=raw.split('>',1)[0]+'>'
    header=header.replace(f'height="{height}"',f'height="{height+350}"').replace(f'0 0 1600 {height}',f'0 0 1600 {height+350}')
    if 'data-design=' not in header:header=header[:-1]+' data-design="bonobono-parody-g">'
    titles={
        'presentation_cover':'날씨를 넣었더니...?!',
        'analysis_journey':'숫자는... 다 같은 숫자일까?',
        'evaluation_boundary':'시험 답안은 보면 안 돼요!!',
        'date_attribution':'백만 행의 날짜를 찾아서~',
        'model_comparison':'전처리를 고쳤는데...?!',
        'weather_model_comparison':'날씨 추가! 결과 발표!!',
        'calibration_tradeoff':'보정하면 무조건 좋을까?',
        'weather_latency':'10분... 기다린 건 아니에요!'}
    # Retain source title/description and the complete scientific content.
    body=raw.split('>',1)[1].rsplit('</svg>',1)[0]
    accessible=''.join(re.findall(r'<(?:title|desc)\b[^>]*>.*?</(?:title|desc)>',body))
    body=re.sub(r'<(?:title|desc)\b[^>]*>.*?</(?:title|desc)>','',body)
    body=body.replace(f'<rect width="1600" height="{height}" fill="{PAPER}"/>',f'<rect width="1600" height="{height}" fill="url(#g-body)"/>',1)
    # WordArt on editorial headings only. Numerical labels and data marks are untouched.
    def tacky_title(match):
        attrs,text=match.groups()
        size=float(re.search(r'font-size="([^"]+)"',attrs).group(1))
        if not 54 <= size <= 86 or re.fullmatch(r'[0-9.,+−% /-]+',text):return match.group(0)
        x=float(re.search(r' x="([^"]+)"', ' '+attrs).group(1)); y=float(re.search(r' y="([^"]+)"', ' '+attrs).group(1))
        from html import unescape
        return wordart(unescape(text),x,y,size,1.3)
    body=re.sub(r'<text ([^>]*font-size="[^"]+"[^>]*)>([^<]*)</text>',tacky_title,body)
    defs='''<defs><linearGradient id="g-rainbow" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#FF3FF2"/><stop offset=".24" stop-color="#FFFC4A"/><stop offset=".5" stop-color="#73FFAE"/><stop offset=".74" stop-color="#49DDFF"/><stop offset="1" stop-color="#B073FF"/></linearGradient><linearGradient id="g-body" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#FFBCEB"/><stop offset=".35" stop-color="#FFF64F"/><stop offset=".7" stop-color="#A1FFD2"/><stop offset="1" stop-color="#B2C3FF"/></linearGradient><pattern id="g-grid" width="48" height="48" patternUnits="userSpaceOnUse"><path d="M48 0 H0 V48" fill="none" stroke="#FFFFFF" stroke-width="3" opacity=".5"/></pattern></defs>'''
    bg=f'<rect width="1600" height="{height+350}" fill="url(#g-rainbow)"/><rect width="1600" height="280" fill="url(#g-grid)"/>'
    stars=''.join(f'<text x="{x}" y="{y}" font-size="{z}" fill="{c}" stroke="#572873" stroke-width="1">★</text>' for x,y,z,c in [(24,78,57,'#FFFF00'),(1160,46,42,'#FF007F'),(1035,229,67,'#F300B3'),(30,229,62,'#008BFF'),(1515,261,50,'#FBFF00')])
    top=wordart(titles[name],95,137,68,-3)+f'<text x="130" y="232" font-size="30" font-family="Noto Serif CJK KR, serif" fill="#23103D" transform="rotate(2 130 232)">☆★ 보노보노 PPT 감성 · G안 · 디자인만 일부러 망했습니다 ★☆</text>'
    mascot=bonobono(1394,124,.72)
    footer=f'<text x="70" y="{height+327}" font-size="30" fill="#241043" font-family="Noto Sans CJK KR, sans-serif">※ 장식은 장난, 수치는 원본 그대로!  |  패러디 실험용 · 공식 캐릭터 자료 아님</text>'
    return '\n'.join([header,accessible,defs,bg,stars,top,mascot,'<g data-evidence-panel="unscaled" transform="translate(0 280)">',body,'</g>',footer,'</svg>'])+'\n'

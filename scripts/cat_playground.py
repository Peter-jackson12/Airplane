"""Original vector cats and decorative scrapbook framing; no external artwork."""
import re
from math import sin, pi
from html import escape

INK='#51413D'; PAPER='#FFF9EF'; PINK='#F4B8BC'; MINT='#C8DDD0'; LAVENDER='#DDD4EC'

def paw(x,y,s=1,rotation=0,color='#D9B8AC'):
    return f'<g aria-hidden="true" transform="translate({x} {y}) rotate({rotation}) scale({s})" fill="{color}"><ellipse cy="9" rx="10" ry="8"/><ellipse cx="-12" cy="-2" rx="4.5" ry="6" transform="rotate(-20 -12 -2)"/><ellipse cx="-4" cy="-10" rx="4.5" ry="6"/><ellipse cx="6" cy="-10" rx="4.5" ry="6"/><ellipse cx="14" cy="-1" rx="4.5" ry="6" transform="rotate(20 14 -1)"/></g>'

def cat(x,y,s=1,color='#E8AF7D',phase=0,flip=False):
    stride=sin(phase)*12; hop=abs(sin(phase))*5
    return f'''<g aria-hidden="true" transform="translate({x} {y-hop}) scale({-s if flip else s} {s})" stroke="{INK}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M-37 3 Q-78 -4 -67 -40 Q-64 -48 -59 -39" fill="none" stroke="{color}" stroke-width="15"/>
    <path d="M-33 14 l{-stride-6} 18 m19 -17 l{stride+3} 18 m25 -16 l{-stride+3} 16 m16 -20 l{stride+5} 18" fill="none" stroke="{color}" stroke-width="14"/>
    <ellipse cx="-12" cy="0" rx="35" ry="24" fill="{color}"/>
    <path d="M5 -12 L2 -48 20 -37 Q30 -41 40 -36 L55 -48 57 -13 Q59 11 32 14 Q8 12 5 -12Z" fill="{color}"/>
    <path d="M10 -39 L12 -23 22 -33 M48 -39 L38 -32 49 -23" fill="#F1BCC0" stroke="none"/>
    <path d="M19 -12 q3 -5 6 0 m15 0 q3 -5 6 0" fill="none"/>
    <path d="M30 -7 l5 0 -2.5 3Z" fill="{INK}" stroke="none"/>
    <path d="M32.5 -4 q-4 6 -8 2 m8 -2 q4 6 8 2 M9 -6 l-13 -3 m14 10 l-14 2 m51 -9 l13 -3 m-14 10 l14 2" fill="none" stroke-width="2"/>
    <path d="M24 -34 l2 8 m7 -10 l0 8 m8 -6 l-2 6" fill="none" opacity=".5"/>
    </g>'''

def frame(parts,height,title):
    """Place an unchanged evidence panel in a roomy decorative frame."""
    root=parts[0]
    newheight=height+230
    root=re.sub(r'height="[^"]+"',f'height="{newheight}"',root,count=1)
    root=re.sub(r'viewBox="[^"]+"',f'viewBox="0 0 1600 {newheight}"',root,count=1)
    out=[root,parts[1],f'<rect width="1600" height="{newheight}" rx="32" fill="{PAPER}"/>',
         '<rect x="22" y="22" width="1556" height="98" rx="28" fill="#F3E6DD"/>',
         f'<text x="78" y="86" fill="{INK}" font-family="sans-serif" font-size="31" font-weight="700">AIRPLANE · CAT PLAYGROUND</text>',
         '<text x="1330" y="84" fill="#77675F" font-family="sans-serif" font-size="25">FIELD NOTES / I</text>',
         '<g transform="translate(0 138)">',*parts[2:],'</g>']
    for x,y,r in [(1110,64,-20),(1165,86,15),(1220,62,-12)]:out.append(paw(x,y,0.7,r))
    out.extend([cat(1375,newheight-43,.55),paw(1195,newheight-43,.7,-30),paw(1250,newheight-28,.7,-10),
                f'<text x="80" y="{newheight-31}" font-family="sans-serif" font-size="22" fill="#77675F">tiny paws, careful evidence.</text>', '</svg>'])
    return '\n'.join(out)+'\n'

def play_frame(t,width=1000,height=280):
    phase=t*2*pi
    x1=480+280*sin(phase+0.8); x2=500+300*sin(phase+pi+0.8)
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 1000 280">',f'<rect width="1000" height="280" rx="28" fill="{PAPER}"/>',
         '<path d="M20 220 Q220 170 460 218 T980 210" fill="none" stroke="#D8DDC5" stroke-width="3" stroke-dasharray="6 9"/>',
         f'<text x="40" y="55" font-family="sans-serif" font-size="21" font-weight="700" fill="{INK}">AIRPLANE / CAT PLAYGROUND</text>',
         '<text x="40" y="89" font-family="sans-serif" font-size="15" fill="#77675F">a little play break between careful experiments</text>',
         '<path d="M800 43 l82 12 -59 19 11 -21 -34 -10Z" fill="#D5D0EA" stroke="#897997" stroke-width="2"/>',
         '<path d="M825 86 Q760 106 774 133" fill="none" stroke="#BEA6AF" stroke-width="2" stroke-dasharray="4 6"/>']
    for i in range(10):out.append(paw(80+i*91,246+8*sin(i),.38,55))
    out.extend([cat(x1,182,.88,'#E8AF7D',phase*6,cos_dir(t+0.8/(2*pi))),cat(x2,194,.72,'#B7C6BF',phase*6+1,not cos_dir(t+0.8/(2*pi))),
                f'<circle cx="{500+70*sin(phase*2):.2f}" cy="{204-30*abs(sin(phase*2)):.2f}" r="13" fill="{PINK}" stroke="#B87D8A" stroke-width="2"/>','</svg>'])
    return '\n'.join(out)

def cos_dir(t):
    from math import cos
    return cos(t*2*pi)<0

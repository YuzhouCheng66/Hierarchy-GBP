"""Render a deterministic mechanism GIF; this is an illustration, not solver data."""
from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
W, H = 1440, 460
S = W / 1120
COLORS = ['#528acb', '#d39448', '#70a483', '#aa8bba']


def font(size, serif=False):
    names = ['georgia.ttf', 'DejaVuSerif.ttf'] if serif else ['segoeui.ttf', 'DejaVuSans.ttf']
    for name in names:
        for path in [Path('C:/Windows/Fonts')/name, Path('/usr/share/fonts/truetype/dejavu')/name]:
            if path.exists():
                return ImageFont.truetype(str(path), round(size*S))
    return ImageFont.load_default()


def ease(x):
    x = max(0, min(1, x))
    return x*x*(3-2*x)


def frame(phase):
    im = Image.new('RGB', (W, H), '#fafbf8')
    d = ImageDraw.Draw(im)
    offset = 40
    def xy(p):
        return (p[0]*S, p[1]*S+offset)
    def line(a, b, color, width=1):
        d.line([xy(a), xy(b)], fill=color, width=max(1, round(width*S)))
    def dot(p, r, color):
        x, y = xy(p); r *= S
        d.ellipse((x-r, y-r, x+r, y+r), fill=color, outline='#ffffff', width=1)
    def text(p, content, size=12, color='#647475', serif=False):
        d.text(xy(p), content, font=font(size, serif), fill=color, anchor='mm')
    hi = 1-.94*ease((phase-.04)/.25)
    lo = 1-.93*ease((phase-.43)/.32)
    def surface(cx, high, low, active):
        def p(x, y):
            z = low*math.sin(2*x+.4)*math.cos(1.7*y)*31+high*math.sin(13*x+8*y)*math.cos(10*y-3*x)*9
            return cx+x*108+y*66, 185+y*37-x*17-z
        for direction in range(2):
            for a in np.linspace(-1, 1, 19):
                ps = [xy(p(a,b) if direction else p(b,a)) for b in np.linspace(-1,1,66)]
                d.line(ps, fill='#c0c8c9', width=1)
        nodes = [p(-.9+x*.3,-.85+y*.425) for y in range(5) for x in range(7)]
        for i, q in enumerate(nodes):
            if i%7: line(nodes[i-1], q, '#a8b3b5')
            if i>=7: line(nodes[i-7], q, '#a8b3b5')
        for i, q in enumerate(nodes):
            dot(q, 3.7, COLORS[int(i%7>=3)+2*int(i>=17)])
            if active and i%7<6 and i+1<len(nodes):
                t=(phase*8+i*.113)%1; a=np.array(q); b=np.array(nodes[i+1])
                dot((1-t)*a+t*b,2.1,'#205fc4')
    surface(212,hi,1,phase<.36)
    surface(908,hi*.3,lo,False)
    def arrow(a,c,b,start,end,dashed=False):
        pts=[]
        for t in np.linspace(0,1,50):
            u=1-t; pts.append(xy((u*u*a[0]+2*u*t*c[0]+t*t*b[0],u*u*a[1]+2*u*t*c[1]+t*t*b[1])))
        if dashed:
            for i in range(0,len(pts)-1,3): d.line(pts[i:i+2],fill='#bd8b80',width=2)
        else: d.line(pts,fill='#8b9fa4',width=2)
        angle=math.atan2(b[1]-c[1],b[0]-c[0])
        for da in [-.45,.45]: line(b,(b[0]-7*math.cos(angle+da),b[1]-7*math.sin(angle+da)),'#8b9fa4',1.4)
        if start<=phase<=end:
            t=(phase-start)/(end-start);u=1-t
            dot((u*u*a[0]+2*u*t*c[0]+t*t*b[0],u*u*a[1]+2*u*t*c[1]+t*t*b[1]),4,'#205fc4')
    arrow((320,121),(362,37),(440,56),.25,.43)
    arrow((510,54),(560,41),(610,54),.43,.7)
    arrow((688,58),(771,37),(806,120),.7,1)
    arrow((382,209),(560,230),(735,209),2,3,True)
    snailx=550+14*phase
    shell=[xy((snailx+(.3+.48*t)*math.cos(t),201+(.3+.48*t)*math.sin(t))) for t in np.arange(0,4*math.pi,.1)]
    d.line(shell,fill='#96766e',width=2)
    for a,b in [((snailx-8,208),(snailx+16,208)),((snailx+16,208),(snailx+18,200)),((snailx+18,200),(snailx+16,196)),((snailx+18,200),(snailx+21,196))]: line(a,b,'#96766e',1.2)
    for cx, distortion in [(475,1),(649,1-ease((phase-.43)/.27))]:
        def point(u,v):
            return (cx+25*(u+distortion*(.24*v+.1*math.sin(math.pi*v)+.12*math.sin(math.pi*u))),
                    63+25*(v+distortion*(.2*u+.3*math.cos(math.pi*u/2)-.12*math.sin(math.pi*v))))
        def curve(a,b,color,width):
            points=[xy(point(a[0]*(1-t)+b[0]*t,a[1]*(1-t)+b[1]*t)) for t in np.linspace(0,1,33)]
            d.line(points,fill=color,width=max(1,round(width*S)))
        for v in np.linspace(-1,1,7):
            curve((-1,v),(1,v),'#dce2e2',.6)
            curve((v,-1),(v,1),'#dce2e2',.6)
        corners=[(-1,-1),(1,-1),(1,1),(-1,1)]
        nodes=[point(u,v) for u,v in corners]
        for a,b in [(0,1),(1,2),(2,3),(3,0),(0,2)]: curve(corners[a],corners[b],'#7e929c',1.3)
        for i,p in enumerate(nodes): dot(p,5.6,COLORS[i])
    text((347,29),'2  Abstraction',15)
    text((562,10),'3  Coarse correction',15)
    text((777,29),'4  Recovery',15)
    text((180,96),'1  GBP smoothing',15,'#205fc4')
    text((560,249),'Pure GBP: slow on low-frequency error',14,'#96766e')
    text((212,270),'Fine graph',17,'#263f49',True)
    text((908,270),'Recovered fine graph',17,'#263f49',True)
    labels=['GBP SMOOTHING: high-frequency error','ABSTRACTION: low-frequency modes','CORRECTION: reduce low-frequency error','RECOVERY: transfer the correction to the fine graph']
    label=labels[min(3,int(phase*4))]
    d.rounded_rectangle((W*.27,H-31,W*.73,H-7),radius=10,fill='#eaf1fb')
    d.text((W/2,H-20),label,font=font(10),fill='#205fc4',anchor='mm')
    return im


def main():
    images=[frame(i/96) for i in range(96)]
    path=ROOT/'code/docs/assets/teaser.gif'
    # A shared palette avoids flickering colors in the animated mesh.
    palette=images[0].quantize(colors=96)
    images=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in images]
    images[0].save(path,save_all=True,append_images=images[1:],duration=115,loop=0,optimize=False,disposal=2)
    print(f'{path}: {path.stat().st_size/1024/1024:.2f} MiB')


if __name__=='__main__':
    main()

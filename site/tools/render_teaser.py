"""Render the standalone white-background README illustration, not a screen capture."""
import argparse
from functools import lru_cache
from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
W, H = 1440, 500
S = W / 940
AA = 3
TYPE_SCALE = 1.5
PLAYBACK_SPEED = 1.5
COLORS = ['#528acb', '#d39448', '#70a483', '#aa8bba']


@lru_cache(maxsize=None)
def font(size, serif=False):
    names = ['georgia.ttf', 'DejaVuSerif.ttf'] if serif else ['segoeui.ttf', 'DejaVuSans.ttf']
    for name in names:
        for path in [Path('C:/Windows/Fonts')/name, Path('/usr/share/fonts/truetype/dejavu')/name]:
            if path.exists():
                return ImageFont.truetype(str(path), round(size*S*AA*TYPE_SCALE))
    return ImageFont.load_default()


def ease(x):
    x = max(0, min(1, x))
    return x*x*(3-2*x)


def frame(phase):
    im = Image.new('RGB', (W*AA, H*AA), '#ffffff')
    d = ImageDraw.Draw(im)
    offset = 28
    def xy(p):
        return (p[0]*S*AA, (p[1]*S+offset)*AA)
    def stroke(points, color, width=1.2):
        d.line(points, fill=color, width=max(1, round(width*S*AA)), joint='curve')
    def line(a, b, color, width=1):
        stroke([xy(a), xy(b)], color, width)
    def dot(p, r, color):
        x, y = xy(p); r *= S*AA
        d.ellipse((x-r, y-r, x+r, y+r), fill=color, outline='#ffffff', width=AA)
    def text(p, content, size=12, color='#647475', serif=False):
        d.text(xy(p), content, font=font(size, serif), fill=color, anchor='mm', align='center')
    hi = 1-.97*ease((phase-.04)/.25)
    lo = 1-.97*ease((phase-.7)/.23)
    def surface(cx, high, low, active):
        def p(x, y):
            z = low*math.sin(2*x+.4)*math.cos(1.7*y)*31+high*math.sin(13*x+8*y)*math.cos(10*y-3*x)*9
            return cx+x*108+y*66, 185+y*37-x*17-z
        for direction in range(2):
            for a in np.linspace(-1, 1, 15):
                ps = [xy(p(a,b) if direction else p(b,a)) for b in np.linspace(-1,1,181)]
                stroke(ps, '#b8c2cb', 1.05)
        for a, b in [((-1,-1),(1,-1)), ((1,-1),(1,1)), ((1,1),(-1,1)), ((-1,1),(-1,-1))]:
            stroke([xy(p(a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])))
                    for t in np.linspace(0,1,181)], '#8e9fac', 1.3)
        coordinates = [(-.9+x*.3,-.85+y*.425) for y in range(5) for x in range(7)]
        nodes = [p(*uv) for uv in coordinates]
        def edge(i,j):
            a,b=coordinates[i],coordinates[j]
            stroke([xy(p(a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])))
                    for t in np.linspace(0,1,33)], '#8c9eac', 1.55)
        for i, q in enumerate(nodes):
            if i%7: edge(i-1,i)
            if i>=7: edge(i-7,i)
        for i, q in enumerate(nodes):
            dot(q, 3.7, COLORS[int(i%7>=3)+2*int(i>=17)])
            if active and i%7<6 and i+1<len(nodes):
                t=(phase*8+i*.113)%1; a=coordinates[i]; b=coordinates[i+1]
                dot(p((1-t)*a[0]+t*b[0],(1-t)*a[1]+t*b[1]),2.1,'#205fc4')
    surface(190,hi,1,phase<.36)
    surface(750,hi*.3,lo,False)
    def arrow(a,c,b,start,end,dashed=False):
        pts=[]
        for t in np.linspace(0,1,50):
            u=1-t; pts.append(xy((u*u*a[0]+2*u*t*c[0]+t*t*b[0],u*u*a[1]+2*u*t*c[1]+t*t*b[1])))
        if dashed:
            for i in range(0,len(pts)-1,3): stroke(pts[i:i+2], '#bd8b80', 1.3)
        else: stroke(pts, '#81959f', 1.5)
        angle=math.atan2(b[1]-c[1],b[0]-c[0])
        for da in [-.45,.45]: line(b,(b[0]-7*math.cos(angle+da),b[1]-7*math.sin(angle+da)),'#8b9fa4',1.4)
        if start<=phase<=end:
            t=(phase-start)/(end-start);u=1-t
            dot((u*u*a[0]+2*u*t*c[0]+t*t*b[0],u*u*a[1]+2*u*t*c[1]+t*t*b[1]),4,'#205fc4')
    arrow((296,121),(333,47),(355,56),.25,.43)
    arrow((425,54),(467,43),(506,54),.43,.7)
    arrow((584,58),(623,48),(650,120),.7,1)
    arrow((366,209),(470,225),(574,209),2,3,True)
    snailx=460+14*phase
    shell=[xy((snailx+(.3+.48*t)*math.cos(t),201+(.3+.48*t)*math.sin(t))) for t in np.arange(0,4*math.pi,.1)]
    stroke(shell, '#96766e', 1.5)
    for a,b in [((snailx-8,208),(snailx+16,208)),((snailx+16,208),(snailx+18,200)),((snailx+18,200),(snailx+16,196)),((snailx+18,200),(snailx+21,196))]: line(a,b,'#96766e',1.2)
    for cx, distortion in [(390,1),(545,1-ease((phase-.43)/.27))]:
        def point(u,v):
            return (cx+25*(u+distortion*(.24*v+.1*math.sin(math.pi*v)+.12*math.sin(math.pi*u))),
                    63+25*(v+distortion*(.2*u+.3*math.cos(math.pi*u/2)-.12*math.sin(math.pi*v))))
        def curve(a,b,color,width):
            points=[xy(point(a[0]*(1-t)+b[0]*t,a[1]*(1-t)+b[1]*t)) for t in np.linspace(0,1,33)]
            stroke(points, color, width)
        for v in np.linspace(-1,1,7):
            curve((-1,v),(1,v),'#c8d1d7',1)
            curve((v,-1),(v,1),'#c8d1d7',1)
        corners=[(-1,-1),(1,-1),(1,1),(-1,1)]
        nodes=[point(u,v) for u,v in corners]
        for a,b in [(0,1),(1,2),(2,3),(3,0),(0,2)]: curve(corners[a],corners[b],'#708795',1.65)
        for i,p in enumerate(nodes): dot(p,5.6,COLORS[i])
    text((270,28),'2  Abstraction',15)
    text((467,8),'3  Coarse correction',15)
    text((670,28),'4  Recovery',15)
    text((145,96),'1  GBP smoothing',15,'#205fc4')
    text((470,242),'Pure GBP:\nslow low-frequency correction',14,'#96766e')
    text((190,255),'Fine graph',17,'#263f49',True)
    text((750,255),'Recovered fine graph',17,'#263f49',True)
    labels=['GBP smooths high-frequency error','Abstraction transfers low-frequency modes',
            'Coarse correction reduces low-frequency error','Recovery transfers the coarse correction']
    stage = 0 if phase < .29 else 1 if phase < .43 else 2 if phase < .7 else 3
    d.rounded_rectangle((W*.18*AA,(H-48)*AA,W*.82*AA,(H-16)*AA),radius=10*AA,fill='#eef4fc')
    d.text((W/2*AA,(H-32)*AA),labels[stage],font=font(12),fill='#205fc4',anchor='mm')
    return im.resize((W,H), Image.Resampling.LANCZOS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', type=Path, help='Optional contact sheet at README display width')
    args = parser.parse_args()
    images=[frame(i/119) for i in range(120)]
    path=ROOT/'code/docs/assets/teaser.gif'
    # Sample every stage for one fixed palette: no temporal dithering or color jumps.
    samples = Image.new('RGB', (360,115*10), 'white')
    for i,index in enumerate(np.linspace(0,len(images)-1,10,dtype=int)):
        samples.paste(images[index].resize((360,115),Image.Resampling.LANCZOS),(0,i*115))
    anchors=COLORS+['#205fc4','#647475','#263f49','#96766e','#8b9fa4',
                    '#81959f','#8e9fac','#8c9eac','#b8c2cb','#bd8b80',
                    '#c8d1d7','#708795','#eef4fc','#ffffff']
    adaptive=samples.quantize(colors=256-len(anchors),dither=Image.Dither.NONE)
    palette=Image.new('P',(1,1))
    palette.putpalette([channel for color in anchors for channel in bytes.fromhex(color[1:])]
                       +adaptive.getpalette()[:(256-len(anchors))*3])
    palette.load()
    white_index=len(anchors)-1
    quantized=[]
    for im in images:
        result=im.quantize(palette=palette,dither=Image.Dither.NONE)
        # Pillow's palette lookup can choose a near-white entry for exact white.
        white=Image.fromarray(np.all(np.asarray(im)==255,axis=2).astype('uint8')*255)
        result.paste(white_index,mask=white)
        quantized.append(result)
    images=quantized
    durations=[80]*len(images)
    durations[0],durations[-1]=350,950
    # GIF durations use 10 ms ticks; quantize boundaries, not each duration.
    boundaries=np.rint(np.cumsum([0]+durations)/(10*PLAYBACK_SPEED)).astype(int)*10
    durations=np.diff(boundaries).tolist()
    images[0].save(path,save_all=True,append_images=images[1:],duration=durations,
                   loop=0,optimize=False,disposal=1,background=white_index)
    preview_frames=[]
    with Image.open(path) as check:
        assert check.size == (W,H) and check.n_frames == len(images)
        for index in range(check.n_frames):
            check.seek(index)
            rgb=check.convert('RGB')
            assert rgb.getpixel((0,0)) == (255,255,255), 'Background must remain pure white'
            assert rgb.crop((0,0,W,12)).getextrema() == ((255,255),)*3, 'Top edge clipped'
            assert rgb.crop((0,H-8,W,H)).getextrema() == ((255,255),)*3, 'Bottom edge clipped'
            assert rgb.crop((0,0,12,H)).getextrema() == ((255,255),)*3, 'Left edge clipped'
            assert rgb.crop((W-12,0,W,H)).getextrema() == ((255,255),)*3, 'Right edge clipped'
            if index in [0,35,66,94,119]:
                preview_frames.append(rgb.resize((830,round(H*830/W)),Image.Resampling.LANCZOS))
        assert 'transparency' not in check.info, 'Use opaque frames, without transparent disposal artifacts'
    if args.preview:
        preview=Image.new('RGB',(830,preview_frames[0].height*len(preview_frames)),'white')
        for row,im in enumerate(preview_frames):
            preview.paste(im,(0,row*im.height))
        args.preview.parent.mkdir(parents=True,exist_ok=True)
        preview.save(args.preview)
    print(f'{path}: {path.stat().st_size/1024/1024:.2f} MiB')
    print('Verified 120 opaque frames, white background, unclipped edges and one stable palette.')
    print(f'Playback: {sum(durations)/1000:.2f} s per loop ({PLAYBACK_SPEED:g}x).')


if __name__=='__main__':
    main()

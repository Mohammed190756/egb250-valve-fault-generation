from PIL import Image
def L(f): return Image.open(f).convert('RGB')
def compose_fig2(out):
    b,a,c=L('p_rig_b2.png'),L('p_drive_a.png'),L('p_valve_c.png')
    gap=200; rw=max(a.width,c.width)
    W=b.width+gap+rw; H=max(b.height, a.height+gap+c.height)
    im=Image.new('RGB',(W,H),'white'); im.paste(b,(0,0))
    im.paste(a,(b.width+gap,0)); im.paste(c,(b.width+gap,a.height+gap))
    im.save(out,dpi=(600,600))
if __name__=='__main__': compose_fig2('Fig2_preview.png')
def compose_fig4(out):
    a,b,c=L('p_sensors_a.png'),L('p_head_b.png'),L('p_daq_c.png')
    gap=200; W=max(a.width, b.width+gap+c.width)
    H=a.height+gap+max(b.height,c.height)
    im=Image.new('RGB',(W,H),'white'); im.paste(a,(0,0)); y=a.height+gap
    im.paste(b,(0,y)); im.paste(c,(b.width+gap,y+(b.height-c.height)//2))
    im.save(out,dpi=(600,600))
if __name__=='__main__': compose_fig4('Fig4_preview.png')

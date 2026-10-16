"""Reproducible private overlays; no medical pixels or identities enter public evidence."""

import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .common import CLASSES, LOCAL, PUBLIC, identity, image_path, save, validation_records
from .analyze import load_predictions, truth
from .metrics import match, paired_recovery


def selections(rows,details,predictions,comparisons=None):
    ranked=sorted(rows,key=lambda r:identity({"seed":42,"purpose":"qualitative","stem":r["filestem"]}))
    criteria={
        "correct_fracture":lambda r:any(b["class_id"]==0 and details[r["filestem"]]["g_to_p"][i]>=0 for i,b in enumerate(r["boxes"])),
        "missed_fracture":lambda r:any(b["class_id"]==0 and details[r["filestem"]]["g_to_p"][i]<0 for i,b in enumerate(r["boxes"])),
        "fracture_iou_50_to_75":lambda r:any(b["class_id"]==0 and x is not None and .5<=x<.75 for b,x in zip(r["boxes"],details[r["filestem"]]["matched_iou"])),
        "false_positive_fracture":lambda r:any(p[5]==0 and p[4]>=.25 and gi<0 for p,gi in zip(predictions[r["filestem"]],details[r["filestem"]]["p_to_g"])),
        "small_missed_periosteal":lambda r:any(b["class_id"]==3 and d["short_768"]<32 and gi<0 for b,d,gi in zip(r["boxes"],details[r["filestem"]]["geometry"],details[r["filestem"]]["g_to_p"])),
        "multiple_nontext_annotations":lambda r:sum(b["class_id"]!=2 for b in r["boxes"])>=4}
    for c in (4,5,6,7):
        criteria[f"missed_{CLASSES[c]}"]=lambda r,c=c:any(b["class_id"]==c and gi<0 for b,gi in zip(r["boxes"],details[r["filestem"]]["g_to_p"]))
    for variant,other in (comparisons or {}).items():
        for event in ("recovered","lost"):
            criteria[f"{variant}_{event}"]=lambda r,other=other,event=event:bool(
                paired_recovery(truth(r),predictions[r["filestem"]],other[r["filestem"]])[event].any())
    result=[]
    for category,predicate in criteria.items():
        eligible=[r for r in ranked if predicate(r)]
        if eligible:result.append((category,eligible[0],len(eligible)))
    return result


def draw_panel(image,gt,pred,title,font):
    canvas=Image.new("RGB",(520,1020),"#101418")
    scale=min(500/image.width,900/image.height)
    resized=image.resize((round(image.width*scale),round(image.height*scale)))
    ox,oy=(520-resized.width)//2,75
    canvas.paste(resized,(ox,oy));draw=ImageDraw.Draw(canvas)
    draw.text((12,12),title,font=font,fill="white")
    draw.text((12,35),"GT green | TP cyan | FP orange; conf >= .25",font=font,fill="white")
    g,p,matrix=match(gt,pred)
    for i,b in enumerate(gt):
        coords=[ox+b[0]*scale,oy+b[1]*scale,ox+b[2]*scale,oy+b[3]*scale]
        draw.rectangle(coords,outline="#47ef76",width=2)
        text=f"G{i} c{int(b[4])}";draw.text((coords[0],max(60,coords[1]-14)),text,font=font,fill="#47ef76",stroke_width=1,stroke_fill="black")
    for i,b in enumerate(pred):
        if b[4]<.25:continue
        coords=[ox+b[0]*scale,oy+b[1]*scale,ox+b[2]*scale,oy+b[3]*scale]
        color="#42ddff" if p[i]>=0 else "#ff9b36"
        draw.rectangle(coords,outline=color,width=2)
        text=f"c{int(b[5])} {b[4]:.2f}"+(f" G{p[i]} IoU{matrix[p[i],i]:.2f}" if p[i]>=0 else " FP")
        draw.text((coords[0],min(970,coords[3]+1)),text,font=font,fill=color,stroke_width=1,stroke_fill="black")
    return canvas


def run():
    rows=validation_records();details=json.loads((LOCAL/"analysis/standard_serial_details.json").read_text())
    preds={v:load_predictions(v) for v in ("standard_serial","sahi_context","sahi_detail")}
    selected=selections(rows,details,preds["standard_serial"],{v:p for v,p in preds.items() if v!="standard_serial"})
    directory=LOCAL/"qualitative";directory.mkdir(parents=True,exist_ok=False)
    font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",13)
    receipt=[];private=[]
    for i,(category,row,eligible) in enumerate(selected,1):
        image=Image.open(image_path(row)).convert("RGB");gt=truth(row)
        sheet=Image.new("RGB",(1560,1020),"black")
        for j,variant in enumerate(preds):sheet.paste(draw_panel(image,gt,preds[variant][row["filestem"]],variant,font),(j*520,0))
        name=f"case_{i:02d}_{category}.png";sheet.save(directory/name)
        receipt.append({"case":i,"category":category,"eligible_images":eligible,"selection":"first eligible in fixed SHA256 order","review":"pending technical visual inspection"})
        private.append({"case":i,"stem":row["filestem"],"patient":row["patient_id"],"image":name})
    save(directory/"selection.json",private)
    save(PUBLIC/"qualitative_selection.json",{"cases":receipt,"private_only":True,"operating_point":{"confidence":.25,"iou":.5},
        "rule":"First eligible per declared category in SHA256(seed=42,purpose=qualitative,stem) order; duplicates between categories retained; no manual replacement",
        "limitations":"Representative diagnostic illustrations, not an exhaustive or clinical annotation review"})


if __name__=="__main__":run()

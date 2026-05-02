"""Custom Gradio CSS."""

CSS = """
:root{
  --bg0:#070a12;
  --bg1:#0a1024;
  --panel: rgba(255,255,255,0.06);
  --panel2: rgba(255,255,255,0.035);
  --stroke: rgba(255,255,255,0.14);
  --stroke2: rgba(255,255,255,0.10);
}

body{
  background:
    radial-gradient(1200px 700px at 12% 10%, rgba(124,58,237,0.20), transparent),
    radial-gradient(900px 600px at 88% 12%, rgba(59,130,246,0.16), transparent),
    radial-gradient(1000px 600px at 50% 100%, rgba(16,185,129,0.10), transparent),
    linear-gradient(180deg, var(--bg0), var(--bg1)) !important;
}

.gradio-container{
  width: 100% !important;
  max-width: 1760px !important;
  margin: 0 auto !important;
  padding: 10px 16px !important;
}

@media (max-width: 900px){
  .gradio-container{ padding: 8px 10px !important; }
}

.block, .wrap{ gap: 8px !important; }
.gr-form{ gap: 6px !important; }
footer, .footer, .gradio-footer, #footer { display:none !important; }

#hero{
  display:flex;
  align-items:center;
  justify-content:center;
  padding: 10px 0 8px;
}

#titlebar{
  width:100%;
  text-align:center;
  font-size: 22px;
  font-weight: 780;
  letter-spacing: 0.4px;
  color: rgba(255,255,255,0.94);
}

.panel{
  border: 1px solid var(--stroke);
  background: linear-gradient(180deg, var(--panel), var(--panel2));
  border-radius: 16px;
  padding: 10px 10px;
  box-shadow: 0 16px 40px rgba(0,0,0,0.28);
  backdrop-filter: blur(10px);
}

.image-panel{
  border: 1px solid var(--stroke2) !important;
  border-radius: 14px !important;
  background: rgba(255,255,255,0.03) !important;
}

.image-panel .gr-image, .image-panel img{ border-radius: 12px !important; }

button.primary{
  background: linear-gradient(90deg, rgba(124,58,237,0.95), rgba(59,130,246,0.95)) !important;
  border: 0 !important;
}

button.secondary{
  border: 1px solid rgba(255,255,255,0.14) !important;
  background: rgba(255,255,255,0.05) !important;
}

#img_in, #img_out{ height: 560px !important; }

@media (max-width: 1100px){
  #img_in, #img_out{ height: 420px !important; }
}

#img_in .gr-image, #img_out .gr-image,
#img_in .image-container, #img_out .image-container{ height: 100% !important; }

#img_in .image-container img, #img_out .image-container img{
  height: 100% !important;
  width: 100% !important;
  object-fit: contain !important;
}

#cmp_before, #cmp_after{ height: 300px !important; }

@media (max-width: 1100px){
  #cmp_before, #cmp_after{ height: 240px !important; }
}

#cmp_before .gr-image, #cmp_after .gr-image,
#cmp_before .image-container, #cmp_after .image-container{ height: 100% !important; }

#cmp_before .image-container img, #cmp_after .image-container img{
  height: 100% !important;
  width: 100% !important;
  object-fit: contain !important;
}
"""

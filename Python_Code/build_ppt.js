const pptxgen = require("pptxgenjs");
const path = require("path");

const ASSETS = "D:/OneDrive/A4_JobREF/9Z_CodeWork/2403CW_GlaucomeClassification/Python_Code/ppt_assets";
const OUT    = "D:/OneDrive/A4_JobREF/9Z_CodeWork/2403CW_GlaucomeClassification/Python_Code/Glaucoma_Classification_Presentation.pptx";

function img(name) { return path.join(ASSETS, name); }

// ─── Palette ──────────────────────────────────────────────────────────────────
const C = {
  teal:   "028090",
  navy:   "065A82",
  mint:   "02C39A",
  dark:   "1E293B",
  gray:   "64748B",
  lgray:  "E2E8F0",
  white:  "FFFFFF",
  coral:  "F96167",
  light:  "F0F9FA",
};

// ─── Shadow factory (fresh object every call) ────────────────────────────────
const sh = () => ({ type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.12 });

let pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author  = "Glaucoma Research Team";
pres.title   = "Glaucoma Classification using CNN";

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 1 — Title
// ═══════════════════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  s.background = { color: C.navy };

  // Top accent strip
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:10, h:0.08, fill:{ color:C.mint }, line:{ color:C.mint } });

  // Large teal circle decoration
  s.addShape(pres.shapes.OVAL, { x:6.8, y:0.3, w:4.5, h:4.5, fill:{ color:"1C7293", transparency:60 }, line:{ color:"1C7293", transparency:60 } });
  s.addShape(pres.shapes.OVAL, { x:7.5, y:1.2, w:3.2, h:3.2, fill:{ color:C.mint, transparency:75 }, line:{ color:C.mint, transparency:75 } });

  s.addText("Glaucoma Fundus Image Classification", {
    x:0.5, y:0.9, w:9.0, h:1.4,
    fontSize:36, fontFace:"Calibri", bold:true, color:C.white, valign:"middle"
  });
  s.addText("Using Convolutional Neural Networks with\nImage Preprocessing Strategies", {
    x:0.5, y:2.3, w:8.2, h:0.9,
    fontSize:18, fontFace:"Calibri", color:C.mint, valign:"middle"
  });
  s.addShape(pres.shapes.LINE, { x:0.5, y:3.25, w:4.5, h:0, line:{ color:C.mint, width:2 } });

  s.addText([
    { text: "Datasets & Methods • Deep Learning Models • Evaluation Results", options: { breakLine: false } }
  ], {
    x:0.5, y:3.5, w:8.5, h:0.4, fontSize:12, color:C.lgray, fontFace:"Calibri"
  });

  s.addText("15 Public Datasets  |  11,031 Training Images  |  5 CNN Models  |  3 Preprocessing Methods", {
    x:0.5, y:4.9, w:9.2, h:0.45,
    fontSize:11, color:"A0C4CC", fontFace:"Calibri", italic:true
  });

  // Slide number
  s.addText("1", { x:9.5, y:5.2, w:0.4, h:0.3, fontSize:9, color:C.gray, align:"right" });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 2 — Section divider: Datasets & Methods
// ═══════════════════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  s.background = { color: C.teal };

  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:10, h:0.06, fill:{ color:C.mint }, line:{ color:C.mint } });
  s.addShape(pres.shapes.OVAL, { x:-0.8, y:3.2, w:5, h:5, fill:{ color:"1C7293", transparency:55 }, line:{ color:"1C7293", transparency:55 } });

  s.addText("SECTION 1", {
    x:0.7, y:1.3, w:8.5, h:0.5,
    fontSize:13, color:C.mint, bold:true, charSpacing:6, fontFace:"Calibri"
  });
  s.addText("Datasets &\nMethods", {
    x:0.7, y:1.8, w:8.5, h:2.0,
    fontSize:48, fontFace:"Calibri", bold:true, color:C.white, valign:"middle"
  });
  s.addText("Dataset sources • Preprocessing • CNN architectures • K-Fold • Evaluation metrics", {
    x:0.7, y:3.9, w:8.5, h:0.5, fontSize:13, color:"CCE9EE", fontFace:"Calibri"
  });
  s.addText("2", { x:9.5, y:5.2, w:0.4, h:0.3, fontSize:9, color:"CCE9EE", align:"right" });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 3 — Dataset Overview (15 sources)
// ═══════════════════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  s.background = { color: C.white };
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:10, h:0.06, fill:{ color:C.teal }, line:{ color:C.teal } });

  s.addText("Dataset Overview", {
    x:0.5, y:0.2, w:9, h:0.6, fontSize:26, fontFace:"Calibri", bold:true, color:C.dark
  });
  s.addText("Combined from 15 publicly available fundus image databases", {
    x:0.5, y:0.75, w:9, h:0.35, fontSize:13, color:C.gray, fontFace:"Calibri"
  });

  // Two-column layout: left = stats cards, right = subdataset bar chart
  // Left column: 3 stat cards
  const cards = [
    { icon:"📊", val:"11,031", label:"Training Images", color:C.teal },
    { icon:"🔬", val:"15",     label:"Source Datasets", color:C.navy },
    { icon:"⚖️", val:"~50/50", label:"Class Balance",   color:C.mint },
  ];
  cards.forEach((c, i) => {
    const cx = 0.3;
    const cy = 1.2 + i * 1.35;
    s.addShape(pres.shapes.RECTANGLE, { x:cx, y:cy, w:3.2, h:1.1, fill:{ color:c.color }, line:{ color:c.color }, shadow:sh() });
    s.addText(c.icon + "  " + c.val, { x:cx+0.15, y:cy+0.1, w:2.9, h:0.5, fontSize:22, fontFace:"Calibri", bold:true, color:C.white, margin:0 });
    s.addText(c.label, { x:cx+0.15, y:cy+0.58, w:2.9, h:0.35, fontSize:11, fontFace:"Calibri", color:"CCE9EE", margin:0 });
  });

  // Right: sub-dataset chart
  s.addImage({ path: img("subdataset_distribution.png"), x:3.7, y:1.1, w:6.0, h:4.1 });

  s.addText("Train split (15 datasets) + separate Test split from RefA1", {
    x:0.5, y:5.1, w:9, h:0.35, fontSize:10, color:C.gray, italic:true, fontFace:"Calibri"
  });
  s.addText("3", { x:9.5, y:5.2, w:0.4, h:0.3, fontSize:9, color:C.gray, align:"right" });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 4 — Data Distribution
// ═══════════════════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  s.background = { color: C.white };
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:10, h:0.06, fill:{ color:C.teal }, line:{ color:C.teal } });

  s.addText("Data Distribution", {
    x:0.5, y:0.2, w:9, h:0.6, fontSize:26, fontFace:"Calibri", bold:true, color:C.dark
  });
  s.addText("Class balance and train/test split composition", {
    x:0.5, y:0.75, w:9, h:0.35, fontSize:13, color:C.gray, fontFace:"Calibri"
  });

  s.addImage({ path: img("dataset_distribution.png"), x:0.4, y:1.15, w:9.2, h:3.9 });

  // Key note
  s.addShape(pres.shapes.RECTANGLE, { x:0.4, y:5.05, w:9.2, h:0.4, fill:{ color:C.light }, line:{ color:C.lgray } });
  s.addText("✓ Dataset is already balanced (no oversampling required)  •  Augmentation: RandomFlip + RandomRotation(±20°) on training folds", {
    x:0.5, y:5.08, w:9.0, h:0.35, fontSize:10, color:C.dark, fontFace:"Calibri", margin:0
  });
  s.addText("4", { x:9.5, y:5.2, w:0.4, h:0.3, fontSize:9, color:C.gray, align:"right" });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 5 — Image Preprocessing Methods
// ═══════════════════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  s.background = { color: C.white };
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:10, h:0.06, fill:{ color:C.teal }, line:{ color:C.teal } });

  s.addText("Image Preprocessing Methods", {
    x:0.5, y:0.2, w:9, h:0.6, fontSize:26, fontFace:"Calibri", bold:true, color:C.dark
  });
  s.addText("All images resized to 224×224 — three enhancement strategies evaluated", {
    x:0.5, y:0.75, w:9, h:0.35, fontSize:13, color:C.gray, fontFace:"Calibri"
  });

  // 3 method cards
  const methods = [
    { name:"Original", desc:"Baseline — resize only.\nNo contrast enhancement.\nPreserves natural fundus appearance.", color:C.navy },
    { name:"Histogram\nEqualization (HE)", desc:"Equalizes luminance channel (YUV).\nGlobal contrast enhancement.\nMay over-amplify noise.", color:C.teal },
    { name:"CLAHE", desc:"Contrast Limited Adaptive\nHistogram Equalization.\nLocalized enhancement,\nreduces over-amplification.", color:C.mint },
  ];
  methods.forEach((m, i) => {
    const cx = 0.3 + i * 3.2;
    s.addShape(pres.shapes.RECTANGLE, { x:cx, y:1.25, w:3.0, h:0.5, fill:{ color:m.color }, line:{ color:m.color } });
    s.addText(m.name, { x:cx, y:1.25, w:3.0, h:0.5, fontSize:13, bold:true, color:C.white, align:"center", valign:"middle", margin:0 });
    s.addShape(pres.shapes.RECTANGLE, { x:cx, y:1.75, w:3.0, h:1.0, fill:{ color:C.light }, line:{ color:C.lgray } });
    s.addText(m.desc, { x:cx+0.1, y:1.78, w:2.8, h:0.95, fontSize:10, color:C.dark, fontFace:"Calibri", valign:"top", margin:0 });
  });

  // Visual comparison — positive
  s.addText("Glaucoma (POSITIVE) Sample:", {
    x:0.5, y:2.85, w:4, h:0.3, fontSize:11, bold:true, color:C.dark, fontFace:"Calibri"
  });
  s.addImage({ path: img("preproc_positive.png"), x:0.3, y:3.15, w:9.4, h:1.2 });

  // Visual comparison — negative
  // s.addText("Normal (NEGATIVE) Sample:", { x:0.5, y:3.25, w:4, h:0.3, fontSize:11, bold:true, color:C.dark });
  // s.addImage({ path: img("preproc_negative.png"), x:0.3, y:3.55, w:9.4, h:1.0 });

  s.addImage({ path: img("preproc_negative.png"), x:0.3, y:4.4, w:9.4, h:1.0 });
  s.addText("Normal (NEGATIVE) Sample:", {
    x:0.5, y:4.38, w:4, h:0.25, fontSize:11, bold:true, color:C.dark, fontFace:"Calibri"
  });

  s.addText("5", { x:9.5, y:5.2, w:0.4, h:0.3, fontSize:9, color:C.gray, align:"right" });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 6 — CNN Models
// ═══════════════════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  s.background = { color: C.white };
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:10, h:0.06, fill:{ color:C.teal }, line:{ color:C.teal } });

  s.addText("CNN Models", {
    x:0.5, y:0.2, w:9, h:0.6, fontSize:26, fontFace:"Calibri", bold:true, color:C.dark
  });
  s.addText("5 pre-trained ImageNet architectures with fine-tuned classification head (2 output classes)", {
    x:0.5, y:0.75, w:9, h:0.35, fontSize:13, color:C.gray, fontFace:"Calibri"
  });

  // Architecture table
  s.addImage({ path: img("model_table.png"), x:0.4, y:1.15, w:9.2, h:2.5 });

  // Training settings row
  s.addShape(pres.shapes.RECTANGLE, { x:0.4, y:3.85, w:9.2, h:1.45, fill:{ color:C.light }, line:{ color:C.lgray }, shadow:sh() });
  s.addText("Training Configuration", {
    x:0.55, y:3.9, w:4, h:0.35, fontSize:12, bold:true, color:C.teal, fontFace:"Calibri", margin:0
  });

  const config = [
    ["Optimizer", "Adam (lr = 1e-4)"],
    ["Loss",      "Cross-Entropy"],
    ["Epochs",    "100 (early stopping)"],
    ["Batch Size","128"],
    ["Input",     "224×224 RGB"],
    ["Framework", "PyTorch 2.x"],
  ];
  config.forEach(([k, v], i) => {
    const col = i < 3 ? 0 : 1;
    const row = i % 3;
    const cx = 0.6 + col * 4.6;
    const cy = 4.3 + row * 0.32;
    s.addText(k + ": ", { x:cx, y:cy, w:1.3, h:0.28, fontSize:10, bold:true, color:C.dark, fontFace:"Calibri", margin:0 });
    s.addText(v,        { x:cx+1.3, y:cy, w:3.0, h:0.28, fontSize:10, color:C.gray, fontFace:"Calibri", margin:0 });
  });

  s.addText("6", { x:9.5, y:5.2, w:0.4, h:0.3, fontSize:9, color:C.gray, align:"right" });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 7 — K-Fold Cross Validation
// ═══════════════════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  s.background = { color: C.white };
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:10, h:0.06, fill:{ color:C.teal }, line:{ color:C.teal } });

  s.addText("K-Fold Cross Validation", {
    x:0.5, y:0.2, w:9, h:0.6, fontSize:26, fontFace:"Calibri", bold:true, color:C.dark
  });
  s.addText("Stratified 5-Fold CV ensures balanced class distribution across all folds", {
    x:0.5, y:0.75, w:9, h:0.35, fontSize:13, color:C.gray, fontFace:"Calibri"
  });

  s.addImage({ path: img("kfold_diagram.png"), x:0.3, y:1.1, w:9.4, h:2.8 });

  // Stats cards
  const kcards = [
    { val:"5",      label:"Folds", color:C.teal },
    { val:"8,824",  label:"Train per Fold", color:C.navy },
    { val:"2,207",  label:"Validation per Fold", color:C.mint },
    { val:"11,031", label:"Total Train Images", color:C.coral },
  ];
  kcards.forEach((c, i) => {
    const cx = 0.35 + i * 2.35;
    s.addShape(pres.shapes.RECTANGLE, { x:cx, y:4.05, w:2.15, h:1.1, fill:{ color:c.color }, line:{ color:c.color }, shadow:sh() });
    s.addText(c.val, { x:cx, y:4.1, w:2.15, h:0.55, fontSize:22, bold:true, color:C.white, align:"center", fontFace:"Calibri", margin:0 });
    s.addText(c.label, { x:cx, y:4.65, w:2.15, h:0.35, fontSize:9, color:"DDEEF0", align:"center", fontFace:"Calibri", margin:0 });
  });

  s.addText("7", { x:9.5, y:5.2, w:0.4, h:0.3, fontSize:9, color:C.gray, align:"right" });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 8 — Evaluation Metrics
// ═══════════════════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  s.background = { color: C.white };
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:10, h:0.06, fill:{ color:C.teal }, line:{ color:C.teal } });

  s.addText("Evaluation Metrics", {
    x:0.5, y:0.2, w:9, h:0.6, fontSize:26, fontFace:"Calibri", bold:true, color:C.dark
  });
  s.addText("Comprehensive metrics for binary glaucoma classification assessment", {
    x:0.5, y:0.75, w:9, h:0.35, fontSize:13, color:C.gray, fontFace:"Calibri"
  });

  // Left: confusion matrix diagram
  s.addText("Confusion Matrix", { x:0.4, y:1.2, w:4.5, h:0.35, fontSize:13, bold:true, color:C.dark, fontFace:"Calibri" });
  s.addImage({ path: img("cm_diagram.png"), x:0.3, y:1.55, w:4.3, h:3.0 });

  // Right: formula cards
  s.addText("Performance Metrics", { x:5.0, y:1.2, w:4.5, h:0.35, fontSize:13, bold:true, color:C.dark, fontFace:"Calibri" });
  s.addImage({ path: img("metrics_formula.png"), x:4.7, y:1.55, w:5.0, h:2.2 });

  // ROC explanation card
  s.addShape(pres.shapes.RECTANGLE, { x:4.7, y:3.85, w:5.0, h:1.3, fill:{ color:C.light }, line:{ color:C.lgray }, shadow:sh() });
  s.addText("ROC Curve & AUC", { x:4.85, y:3.9, w:4.7, h:0.35, fontSize:12, bold:true, color:C.teal, fontFace:"Calibri", margin:0 });
  s.addText([
    { text: "• TPR (Sensitivity) vs FPR (1−Specificity) at all thresholds", options: { breakLine: true } },
    { text: "• AUC = 1.0 → perfect classifier", options: { breakLine: true } },
    { text: "• AUC = 0.5 → random classifier", options: { breakLine: false } },
  ], { x:4.85, y:4.3, w:4.7, h:0.8, fontSize:10, color:C.dark, fontFace:"Calibri", margin:0 });

  s.addText("8", { x:9.5, y:5.2, w:0.4, h:0.3, fontSize:9, color:C.gray, align:"right" });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 9 — Section divider: Results
// ═══════════════════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  s.background = { color: C.dark };

  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:10, h:0.06, fill:{ color:C.coral }, line:{ color:C.coral } });
  s.addShape(pres.shapes.OVAL, { x:6.5, y:1.5, w:5, h:5, fill:{ color:"2D3F55", transparency:40 }, line:{ color:"2D3F55", transparency:40 } });

  s.addText("SECTION 2", {
    x:0.7, y:1.3, w:8.5, h:0.5,
    fontSize:13, color:C.coral, bold:true, charSpacing:6, fontFace:"Calibri"
  });
  s.addText("Results", {
    x:0.7, y:1.8, w:8.5, h:1.8,
    fontSize:60, fontFace:"Calibri", bold:true, color:C.white, valign:"middle"
  });
  s.addText("Preprocessing effects • Model accuracy & AUC • Confusion matrices • ROC curves", {
    x:0.7, y:3.7, w:8.5, h:0.5, fontSize:13, color:"9BB8C9", fontFace:"Calibri"
  });
  s.addText("9", { x:9.5, y:5.2, w:0.4, h:0.3, fontSize:9, color:C.gray, align:"right" });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 10 — Preprocessing Visual Results
// ═══════════════════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  s.background = { color: C.white };
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:10, h:0.06, fill:{ color:C.coral }, line:{ color:C.coral } });

  s.addText("Preprocessing: Visual Results", {
    x:0.5, y:0.2, w:9, h:0.6, fontSize:26, fontFace:"Calibri", bold:true, color:C.dark
  });
  s.addText("Effect of three enhancement methods on representative fundus images", {
    x:0.5, y:0.75, w:9, h:0.35, fontSize:13, color:C.gray, fontFace:"Calibri"
  });

  s.addText("Glaucoma (POSITIVE)", {
    x:0.4, y:1.15, w:9, h:0.3, fontSize:12, bold:true, color:C.teal, fontFace:"Calibri"
  });
  s.addImage({ path: img("preproc_positive.png"), x:0.3, y:1.45, w:9.4, h:1.85 });

  s.addText("Normal (NEGATIVE)", {
    x:0.4, y:3.35, w:9, h:0.3, fontSize:12, bold:true, color:C.navy, fontFace:"Calibri"
  });
  s.addImage({ path: img("preproc_negative.png"), x:0.3, y:3.62, w:9.4, h:1.65 });

  s.addText("10", { x:9.4, y:5.2, w:0.5, h:0.3, fontSize:9, color:C.gray, align:"right" });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 11 — Model × Preprocessing Performance
// ═══════════════════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  s.background = { color: C.white };
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:10, h:0.06, fill:{ color:C.coral }, line:{ color:C.coral } });

  s.addText("Model Performance Comparison", {
    x:0.5, y:0.2, w:9, h:0.6, fontSize:26, fontFace:"Calibri", bold:true, color:C.dark
  });
  s.addText("Validation Accuracy & AUC across all 5 models × 3 preprocessing methods (5-fold CV)", {
    x:0.5, y:0.75, w:9, h:0.35, fontSize:13, color:C.gray, fontFace:"Calibri"
  });

  s.addImage({ path: img("model_comparison.png"), x:0.3, y:1.1, w:9.4, h:3.95 });

  // Key finding badge
  s.addShape(pres.shapes.RECTANGLE, { x:0.4, y:5.05, w:9.2, h:0.4, fill:{ color:"FFF5E6" }, line:{ color:"FFD699" } });
  s.addText("★  Best overall: VGG-16 with Original preprocessing — Accuracy 91.8%, AUC 0.973", {
    x:0.55, y:5.08, w:9.0, h:0.35, fontSize:11, bold:true, color:"7C4D00", fontFace:"Calibri", margin:0
  });
  s.addText("11", { x:9.4, y:5.2, w:0.5, h:0.3, fontSize:9, color:C.gray, align:"right" });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 12 — Heatmap
// ═══════════════════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  s.background = { color: C.white };
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:10, h:0.06, fill:{ color:C.coral }, line:{ color:C.coral } });

  s.addText("Performance Heatmap", {
    x:0.5, y:0.2, w:9, h:0.6, fontSize:26, fontFace:"Calibri", bold:true, color:C.dark
  });
  s.addText("Preprocessing × Model matrix — Accuracy (left) and AUC (right)", {
    x:0.5, y:0.75, w:9, h:0.35, fontSize:13, color:C.gray, fontFace:"Calibri"
  });

  s.addImage({ path: img("heatmap.png"), x:0.3, y:1.1, w:9.4, h:3.2 });

  // Observations
  s.addShape(pres.shapes.RECTANGLE, { x:0.4, y:4.4, w:9.2, h:0.85, fill:{ color:C.light }, line:{ color:C.lgray }, shadow:sh() });
  s.addText([
    { text: "Key Observations:", options: { bold: true, breakLine: true } },
    { text: "• VGG-16 achieves top accuracy across all preprocessing variants (91.8% / 90.3% / 90.1%)", options: { breakLine: true } },
    { text: "• All models perform strongly (88.6%–91.8%) — Original preprocessing consistently yields the best results", options: { breakLine: false } },
  ], {
    x:0.55, y:4.43, w:9.0, h:0.8, fontSize:10, color:C.dark, fontFace:"Calibri", margin:0
  });
  s.addText("12", { x:9.4, y:5.2, w:0.5, h:0.3, fontSize:9, color:C.gray, align:"right" });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 13 — Sensitivity vs Specificity Trade-off
// ═══════════════════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  s.background = { color: C.white };
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:10, h:0.06, fill:{ color:C.coral }, line:{ color:C.coral } });

  s.addText("Sensitivity vs. Specificity — Best Models", {
    x:0.5, y:0.2, w:9, h:0.6, fontSize:26, fontFace:"Calibri", bold:true, color:C.dark
  });
  s.addText("VGG-16 results across all three preprocessing strategies", {
    x:0.5, y:0.75, w:9, h:0.35, fontSize:13, color:C.gray, fontFace:"Calibri"
  });

  s.addImage({ path: img("sens_spec.png"), x:0.8, y:1.1, w:8.4, h:3.75 });

  s.addShape(pres.shapes.RECTANGLE, { x:0.4, y:4.98, w:9.2, h:0.4, fill:{ color:C.light }, line:{ color:C.lgray } });
  s.addText("Clinical note: High Sensitivity critical for glaucoma screening (avoid missed diagnoses). CLAHE VGG-16 gives best Specificity (94.4%).", {
    x:0.55, y:5.0, w:9.0, h:0.38, fontSize:10, color:C.dark, fontFace:"Calibri", italic:true, margin:0
  });
  s.addText("13", { x:9.4, y:5.2, w:0.5, h:0.3, fontSize:9, color:C.gray, align:"right" });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 14 — Confusion Matrices (VGG-16, 3 preprocessing)
// ═══════════════════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  s.background = { color: C.white };
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:10, h:0.06, fill:{ color:C.coral }, line:{ color:C.coral } });

  s.addText("Confusion Matrices — VGG-16", {
    x:0.5, y:0.2, w:9, h:0.6, fontSize:26, fontFace:"Calibri", bold:true, color:C.dark
  });
  s.addText("Best model (VGG-16) evaluated across three preprocessing methods", {
    x:0.5, y:0.75, w:9, h:0.35, fontSize:13, color:C.gray, fontFace:"Calibri"
  });

  const labels = ["Original", "HE", "CLAHE"];
  const cms    = ["cm_orig_vgg16.png", "cm_he_vgg16.png", "cm_clahe_vgg16.png"];
  const accs   = ["91.8%", "90.3%", "90.1%"];
  const colors = [C.teal, C.navy, C.mint];

  labels.forEach((lbl, i) => {
    const cx = 0.25 + i * 3.2;
    s.addShape(pres.shapes.RECTANGLE, { x:cx, y:1.2, w:3.0, h:0.38, fill:{ color:colors[i] }, line:{ color:colors[i] } });
    s.addText(lbl + "  (" + accs[i] + " Acc)", { x:cx, y:1.2, w:3.0, h:0.38, fontSize:11, bold:true, color:C.white, align:"center", valign:"middle", margin:0 });
    s.addImage({ path: img(cms[i]), x:cx, y:1.58, w:3.0, h:3.6 });
  });

  s.addText("14", { x:9.4, y:5.2, w:0.5, h:0.3, fontSize:9, color:C.gray, align:"right" });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 15 — ROC Curves (VGG-16, 3 preprocessing)
// ═══════════════════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  s.background = { color: C.white };
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:10, h:0.06, fill:{ color:C.coral }, line:{ color:C.coral } });

  s.addText("ROC Curves — VGG-16", {
    x:0.5, y:0.2, w:9, h:0.6, fontSize:26, fontFace:"Calibri", bold:true, color:C.dark
  });
  s.addText("Receiver Operating Characteristic curves — best model per preprocessing variant", {
    x:0.5, y:0.75, w:9, h:0.35, fontSize:13, color:C.gray, fontFace:"Calibri"
  });

  const rocs  = ["roc_orig_vgg16.png", "roc_he_vgg16.png", "roc_clahe_vgg16.png"];
  const names = ["Original", "HE", "CLAHE"];
  const aucs  = ["0.973", "0.963", "0.964"];
  const col2  = [C.teal, C.navy, C.mint];

  names.forEach((n, i) => {
    const cx = 0.25 + i * 3.2;
    s.addShape(pres.shapes.RECTANGLE, { x:cx, y:1.2, w:3.0, h:0.38, fill:{ color:col2[i] }, line:{ color:col2[i] } });
    s.addText(n + "  (AUC " + aucs[i] + ")", { x:cx, y:1.2, w:3.0, h:0.38, fontSize:11, bold:true, color:C.white, align:"center", valign:"middle", margin:0 });
    s.addImage({ path: img(rocs[i]), x:cx, y:1.58, w:3.0, h:3.6 });
  });

  s.addText("15", { x:9.4, y:5.2, w:0.5, h:0.3, fontSize:9, color:C.gray, align:"right" });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 16 — Summary Table (all results)
// ═══════════════════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  s.background = { color: C.white };
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:10, h:0.06, fill:{ color:C.teal }, line:{ color:C.teal } });

  s.addText("Results Summary", {
    x:0.5, y:0.2, w:9, h:0.6, fontSize:26, fontFace:"Calibri", bold:true, color:C.dark
  });

  const rows = [
    [{ text:"Preprocessing", options:{ bold:true, fill:{ color:C.teal }, color:C.white } },
     { text:"Model", options:{ bold:true, fill:{ color:C.teal }, color:C.white } },
     { text:"Accuracy", options:{ bold:true, fill:{ color:C.teal }, color:C.white } },
     { text:"AUC", options:{ bold:true, fill:{ color:C.teal }, color:C.white } },
     { text:"Sensitivity", options:{ bold:true, fill:{ color:C.teal }, color:C.white } },
     { text:"Specificity", options:{ bold:true, fill:{ color:C.teal }, color:C.white } }],
    ["Original", "AlexNet",      "89.9%", "0.964", "87.9%", "91.9%"],
    [{ text:"Original", options:{ fill:{ color:"FFF8E6" } } }, { text:"VGG-16 ★", options:{ bold:true, fill:{ color:"FFF8E6" } } }, { text:"91.8%", options:{ bold:true, fill:{ color:"FFF8E6" } } }, { text:"0.973", options:{ bold:true, fill:{ color:"FFF8E6" } } }, "89.6%", "94.0%"],
    ["Original", "DenseNet-121", "91.6%", "0.973", "92.4%", "90.7%"],
    ["Original", "ResNet-18",    "91.3%", "0.971", "90.4%", "92.3%"],
    ["Original", "MobileNet-V2", "90.2%", "0.961", "86.3%", "94.1%"],
    ["HE",       "AlexNet",      "88.7%", "0.953", "87.1%", "90.3%"],
    [{ text:"HE", options:{ fill:{ color:"F0FAFF" } } }, { text:"VGG-16 ★", options:{ bold:true, fill:{ color:"F0FAFF" } } }, { text:"90.3%", options:{ bold:true, fill:{ color:"F0FAFF" } } }, { text:"0.963", options:{ bold:true, fill:{ color:"F0FAFF" } } }, "88.5%", "92.1%"],
    ["HE",       "DenseNet-121", "89.3%", "0.960", "86.3%", "92.4%"],
    ["HE",       "ResNet-18",    "88.8%", "0.953", "86.9%", "90.7%"],
    ["HE",       "MobileNet-V2", "89.1%", "0.955", "85.4%", "92.8%"],
    ["CLAHE",    "AlexNet",      "88.9%", "0.956", "86.7%", "91.2%"],
    [{ text:"CLAHE", options:{ fill:{ color:"F5FFF8" } } }, { text:"VGG-16 ★", options:{ bold:true, fill:{ color:"F5FFF8" } } }, { text:"90.1%", options:{ bold:true, fill:{ color:"F5FFF8" } } }, { text:"0.964", options:{ bold:true, fill:{ color:"F5FFF8" } } }, "85.8%", "94.4%"],
    ["CLAHE",    "DenseNet-121", "89.6%", "0.963", "85.1%", "94.2%"],
    ["CLAHE",    "ResNet-18",    "89.8%", "0.959", "87.1%", "92.6%"],
    ["CLAHE",    "MobileNet-V2", "89.6%", "0.958", "85.6%", "93.6%"],
  ];

  s.addTable(rows, {
    x:0.3, y:0.88, w:9.4,
    border: { pt:0.5, color:"DDDDDD" },
    fontSize:9.5,
    fontFace:"Calibri",
    colW:[1.3,1.5,1.3,1.1,1.4,1.4],
    align:"center",
    valign:"middle",
  });

  s.addText("16", { x:9.4, y:5.2, w:0.5, h:0.3, fontSize:9, color:C.gray, align:"right" });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 17 — Conclusion
// ═══════════════════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  s.background = { color: C.navy };
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:10, h:0.06, fill:{ color:C.mint }, line:{ color:C.mint } });
  s.addShape(pres.shapes.OVAL, { x:6.5, y:2.5, w:5, h:5, fill:{ color:"1C7293", transparency:60 }, line:{ color:"1C7293", transparency:60 } });

  s.addText("Conclusions", {
    x:0.5, y:0.3, w:9, h:0.7, fontSize:32, fontFace:"Calibri", bold:true, color:C.white
  });

  const points = [
    { icon:"🏆", title:"Best Model",           desc:"VGG-16 with Original preprocessing achieves the highest accuracy (91.8%) and AUC (0.973)" },
    { icon:"🎨", title:"Preprocessing Impact", desc:"Original images outperform HE and CLAHE. CLAHE VGG-16 achieves best Specificity (94.4%) with slight accuracy trade-off" },
    { icon:"⚖️", title:"Model Consistency",    desc:"All 5 models perform strongly (88.6%–91.8%) — no catastrophic failures; Run1 training generalises well across architectures" },
    { icon:"📈", title:"AUC Consistency",       desc:"VGG-16 maintains AUC > 0.96 across all preprocessing variants — highly reliable for clinical screening" },
    { icon:"🔍", title:"Future Work",           desc:"Ensemble methods, GAN-based augmentation, and explainability (Grad-CAM) for clinical adoption" },
  ];

  points.forEach((p, i) => {
    const cy = 1.1 + i * 0.88;
    s.addShape(pres.shapes.RECTANGLE, { x:0.4, y:cy, w:9.2, h:0.78, fill:{ color:"1C3A5A", transparency:30 }, line:{ color:"FFFFFF", transparency:70 } });
    s.addText(p.icon + "  " + p.title + ":", { x:0.6, y:cy+0.05, w:2.5, h:0.35, fontSize:11, bold:true, color:C.mint, fontFace:"Calibri", margin:0 });
    s.addText(p.desc, { x:0.6, y:cy+0.38, w:8.8, h:0.35, fontSize:10, color:C.white, fontFace:"Calibri", margin:0 });
  });

  s.addText("Thank you", {
    x:0.5, y:5.1, w:4, h:0.4, fontSize:13, color:C.mint, italic:true, fontFace:"Calibri"
  });
  s.addText("17", { x:9.4, y:5.2, w:0.5, h:0.3, fontSize:9, color:C.gray, align:"right" });
}

// ─────────────────────────────────────────────────────────────────────────────
pres.writeFile({ fileName: OUT })
  .then(() => console.log("✅  Saved:", OUT))
  .catch(err => console.error("❌  Error:", err));

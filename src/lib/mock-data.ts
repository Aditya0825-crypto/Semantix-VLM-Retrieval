// Mock demo data shared between retrieval and embeddings pages.
export type ImageItem = {
  id: string;
  url: string;
  caption: string;
  tags: string[];
};

// Curated Unsplash imagery (stable IDs)
export const IMAGE_LIBRARY: ImageItem[] = [
  { id: "i1", url: "https://images.unsplash.com/photo-1518770660439-4636190af475?w=800&q=80", caption: "Macro shot of a circuit board with blue conductive traces.", tags: ["technology","electronics","blue"] },
  { id: "i2", url: "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=800&q=80", caption: "Matrix-style green code raining down a dark screen.", tags: ["code","cyber","matrix"] },
  { id: "i3", url: "https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=800&q=80", caption: "A friendly white robot arm reaching out in a studio.", tags: ["robot","ai","studio"] },
  { id: "i4", url: "https://images.unsplash.com/photo-1531297484001-80022131f5a1?w=800&q=80", caption: "Surface of a laptop with glowing keys in a dim room.", tags: ["laptop","glow","workspace"] },
  { id: "i5", url: "https://images.unsplash.com/photo-1488229297570-58520851e868?w=800&q=80", caption: "Modern data center server racks with cyan accent lights.", tags: ["servers","datacenter","infrastructure"] },
  { id: "i6", url: "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=800&q=80", caption: "Earth at night seen from orbit, lit by city lights.", tags: ["space","earth","satellite"] },
  { id: "i7", url: "https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=800&q=80", caption: "Abstract neural network of glowing nodes and edges.", tags: ["neural","network","abstract"] },
  { id: "i8", url: "https://images.unsplash.com/photo-1535378917042-10a22c95931a?w=800&q=80", caption: "Holographic interface projecting data above a hand.", tags: ["hologram","ui","future"] },
  { id: "i9", url: "https://images.unsplash.com/photo-1518709268805-4e9042af2176?w=800&q=80", caption: "Glowing fiber optic cables fanning out from a hub.", tags: ["fiber","light","network"] },
  { id: "i10", url: "https://images.unsplash.com/photo-1581090464777-f3220bbe1b8b?w=800&q=80", caption: "Engineer interacting with floating holographic schematics.", tags: ["engineer","hologram","ai"] },
  { id: "i11", url: "https://images.unsplash.com/photo-1517694712202-14dd9538aa97?w=800&q=80", caption: "Developer typing on a mechanical keyboard at night.", tags: ["developer","keyboard","night"] },
  { id: "i12", url: "https://images.unsplash.com/photo-1542831371-29b0f74f9713?w=800&q=80", caption: "Lines of source code on a high-contrast dark terminal.", tags: ["code","terminal","dark"] },
];

export const EXAMPLE_QUERIES = [
  "A futuristic AI assistant in a glowing interface",
  "Quantum computer in a research lab",
  "Neural network visualization at night",
  "Robot hand holding a small planet",
  "Cyberpunk city skyline in the rain",
  "Holographic data flowing through space",
];

// Deterministic-ish "embedding" sim — returns ranked results for a text query.
export function searchImagesByText(query: string, topK = 8): { item: ImageItem; sim: number }[] {
  const q = query.toLowerCase();
  const tokens = q.split(/\s+/).filter(Boolean);
  const scored = IMAGE_LIBRARY.map((item) => {
    const text = (item.caption + " " + item.tags.join(" ")).toLowerCase();
    let score = 0;
    for (const t of tokens) {
      if (!t) continue;
      if (text.includes(t)) score += 1;
      if (item.tags.some((tag) => tag.includes(t))) score += 0.5;
    }
    // baseline pseudo-similarity so we always show something
    const hash = [...q].reduce((a, c) => a + c.charCodeAt(0), 0);
    const noise = ((hash * (parseInt(item.id.slice(1)) + 7)) % 100) / 400;
    const sim = Math.min(0.99, 0.55 + score * 0.08 + noise);
    return { item, sim };
  })
    .sort((a, b) => b.sim - a.sim)
    .slice(0, topK);
  return scored;
}

export function captionsForImage(_url: string, topK = 8): { caption: string; sim: number }[] {
  // Random-ish but stable set per call session
  const pool = [
    "A high-resolution photograph showing intricate technological detail.",
    "An abstract digital composition with strong neon highlights.",
    "A scene depicting human–AI collaboration in a futuristic setting.",
    "A close-up macro view emphasizing texture and light.",
    "A wide-angle shot of a luminous interface against dark surroundings.",
    "An artistic rendering blending photography and 3D-generated elements.",
    "A studio composition with strong rim lighting and a single subject.",
    "A documentary-style frame capturing a moment in a tech environment.",
  ];
  return pool
    .map((caption, i) => ({ caption, sim: 0.95 - i * 0.07 - Math.random() * 0.03 }))
    .sort((a, b) => b.sim - a.sim)
    .slice(0, topK);
}

from PIL import Image
for i in range(1,35):
    inpt = fr"C:\Users\User\File\vmav{i}.jpg"
    base = Image.open(inpt).convert("RGBA")
    cell_w, cell_h = base.size
    even_row_tile = base.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_180)
    cols = 8
    rows = 6
    out_w = cols * cell_w
    out_h = rows * cell_h
    bg = "white"
    out = Image.new("RGBA", (out_w, out_h), bg)
    for r in range(rows):
        tile = base if (r % 2 == 0) else even_row_tile
        for c in range(cols):
            x = c * (cell_w)
            y = r * (cell_h)
            out.paste(tile, (x, y), tile)
    out = out.convert("RGB")
    outpt = fr"C:\Users\Mathijs Born\Downloads\vmav{i}_array.jpg"
    out.save(outpt)

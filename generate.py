import math
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import markdown
from lxml import etree


def rgb_from_hue(x: float) -> str:
    hue = (x % 1) * 360.0
    lightness = 0.5

    c = lightness * (1 - abs(2 * lightness - 1))
    x = c * (1 - abs((hue / 60) % 2 - 1))
    m = lightness - c / 2

    if 0 <= hue < 60:
        r, g, b = c, x, 0
    elif 60 <= hue < 120:
        r, g, b = x, c, 0
    elif 120 <= hue < 180:
        r, g, b = 0, c, x
    elif 180 <= hue < 240:
        r, g, b = 0, x, c
    elif 240 <= hue < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)

    return f"#{r:02x}{g:02x}{b:02x}"


def spiral_dist(row: int, col: int, rows: int, cols: int) -> int:
    center_row = rows / 2
    center_col = cols / 2

    dx = col - center_col
    dy = row - center_row

    radius = math.sqrt(dx**2 + dy**2)
    theta = math.atan2(dy, dx)

    if theta < 0:
        theta += 2 * math.pi

    spiral_value = round((theta / (2 * math.pi) * 5) + radius)

    return spiral_value


class Table:
    def __init__(self, cols: int, rows: int):
        self.cols = cols
        self.rows = rows

    def to_element(self) -> etree._Element:
        table = etree.Element("table")
        table.set("width", "100%")
        for row in range(self.rows):
            tr = etree.Element("tr")
            for col in range(self.cols):
                dist = round(
                    math.sqrt((row - self.rows / 2) ** 2 + (col - self.cols / 2) ** 2)
                )
                spir = spiral_dist(row, col, self.rows, self.cols)
                color = rgb_from_hue(dist / 10)

                td = etree.Element("td")
                td.set("height", "30")
                td.set("cellpadding", "0")
                td.set("cellspacing", "0")
                td.set("border", "0")
                marquee = etree.Element("marquee")
                marquee.set("scrollamount", "0")
                marquee.set("truespeed", "true")
                marquee.set("height", "30")
                marquee2 = etree.Element("marquee")
                marquee2.set("behavior", "scroll")
                marquee2.set("scrollamount", str(4))
                marquee2.set("scrolldelay", str(80))
                marquee2.set(
                    "direction", ["left", "right", "up", "down"][(row + col) % 2]
                )
                marquee2.set("truespeed", "true")
                marquee2.set("height", "20")
                marquee2.set("width", str(int(10)))
                marquee.append(marquee2)

                font = etree.Element("font")
                font.set("color", color)
                max_dist = 6
                spir = spir % max_dist
                nbsp = "\xa0"
                content = "*"
                if (row + col) % 2 == 0:
                    nbsp_before = spir
                    nbsp_after = max_dist - spir
                else:
                    nbsp_before = max_dist - spir
                    nbsp_after = spir
                font.text = (nbsp * nbsp_before + content + nbsp * nbsp_after) * 4
                marquee2.append(font)

                code = etree.Element("code")
                code.set("family", "monospace")
                code.set("width", "20")
                code.append(marquee)
                td.append(code)
                tr.append(td)
            table.append(tr)
        return table

    def __str__(self) -> str:
        return etree.tostring(self.to_element(), pretty_print=True, encoding="unicode")


def generate_html() -> str:
    """Generate the complete HTML page with marquee grid pattern."""
    template_dir = Path("templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("base.html")

    with open(template_dir / "content.md", "r", encoding="utf-8") as f:
        markdown_content = f.read()
    html_content = markdown.markdown(markdown_content)

    table = Table(30, 20)

    rendered_html = template.render(
        title="Marquee Grid Pattern",
        table=table,
        main_content=html_content,
    )

    return rendered_html


def main():
    html_content = generate_html()

    public_dir = Path("public")
    public_dir.mkdir(exist_ok=True)
    index_html = public_dir / "index.html"
    index_html.write_text(html_content)


if __name__ == "__main__":
    main()

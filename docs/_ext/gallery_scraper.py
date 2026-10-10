from sphinx_gallery.scrapers import matplotlib_scraper


class GalleryScraper:
    """Saves gallery figures at a web resolution, cropped to their content."""

    def __repr__(self):
        return "GalleryScraper"

    def __call__(self, *args, **kwargs):
        return matplotlib_scraper(*args, dpi=150, bbox_inches="tight", **kwargs)

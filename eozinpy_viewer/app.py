"""A simple WSI viewer written in Python.

Copyright 2025 @yujota
License: MIT
"""
import argparse
import os
import math

import pyglet


class Model():
    def __init__(self, window_height):
        """Model of slide position and scale

        :param window_height: height of app window
        :type window_height: int
        """
        self.sx = 0  # Slide's far left position from window left
        self.sy = 0  # Slide's far top position from window top
        self.w_height = window_height
        self.lm_scale = 1.0  # scale factor of lowest mag image
        self.zoom_factor = math.sqrt(2)

    def update_window_size(self, width, height):
        self.w_height = height

    def update_move(self, x, y, dx, dy):
        self.sx = self.sx + dx
        self.sy = self.sy - dy

    def update_scale(self, focus_x, focus_y, zoom_in=True):
        if zoom_in:
            next_scale = self.lm_scale * self.zoom_factor
            zf = self.zoom_factor
        else:
            next_scale = self.lm_scale / self.zoom_factor
            zf = 1 / self.zoom_factor
        focus_sx, focus_sy = Coord.pyg2slide(focus_x, focus_y, self.w_height)
        diff_sx = (self.sx - focus_sx) * zf
        diff_sy = (self.sy - focus_sy) * zf
        new_sx = focus_sx + diff_sx
        new_sy = focus_sy + diff_sy
        self.lm_scale = next_scale
        self.sx = new_sx
        self.sy = new_sy


class View():
    def __init__(self, reader, window_height):
        """View class 

        :param reader: abstructed reader class
        :type reader: PathologyReader
        :param window_height: height of app window
        :type window_height: int
        """
        self.batch = pyglet.graphics.Batch()
        self.bg_batch = pyglet.graphics.Batch()
        p_img = convert_pil_image_to_pyglet_image(reader.lowest_mg_img)
        x, y = Coord.slide2pyg(
                sx=0, 
                sy=0, 
                w_height=window_height, 
                img_height=reader.lowest_mg_dim[1])
        self.lowest_mag = pyglet.sprite.Sprite(p_img, x, y, batch=self.bg_batch)
        self.reader = reader
        self.sprites = dict()

    def update(self, model, grid):
        """Update view by given model and tile_grid

        :type model: Model
        :type grid: TileGrid
        """
        self.lowest_mag.scale = model.lm_scale
        x, y = Coord.slide2pyg(
                sx=model.sx, 
                sy=model.sy, 
                w_height=model.w_height,
                img_height=self.lowest_mag.height)
        self.lowest_mag.x = x
        self.lowest_mag.y = y
        reqs = grid.require(model.sx, model.sy, model.lm_scale)
        res = self.reader.request(reqs - self.sprites.keys())
        new_sprites = dict()
        for r in reqs:
            lv, row, col = r
            sprite = self.sprites.get(r, None)
            if sprite is None:
                img = res.get(r, None)
                if img is None:
                    continue
                sprite = pyglet.sprite.Sprite(img, 0, 0, batch=self.batch)
            t_sx, t_sy, img_scale = grid.draw_position(
                    model.sx, model.sy, model.lm_scale, r)
            sprite.scale = img_scale
            t_x, t_y = Coord.slide2pyg(
                    t_sx, t_sy, grid.w_height, img_height=sprite.height)
            sprite.x = t_x
            sprite.y = t_y
            new_sprites[r] = sprite
        for k, s in self.sprites.items():
            if k not in new_sprites.keys():
                del s
        self.sprites = new_sprites

    def on_draw(self):
        self.bg_batch.draw()
        self.batch.draw()


class EventHandler():
    def __init__(self, model, view, grid):
        """Event handler dispatched by pyglet window

        :type model: Model
        :type view: View
        :type grid: TileGrid
        """
        self.model = model
        self.view = view
        self.grid = grid

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.model.update_move(x, y, dx, dy)
        self.view.update(self.model, self.grid)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.model.update_scale(x, y, zoom_in=scroll_y > 0)
        self.view.update(self.model, self.grid)

    def on_resize(self, width, height):
        self.model.update_window_size(width, height)
        self.grid.update_window_size(width, height)
        self.view.update(self.model, self.grid)


# Helpers
class PathologyReader():
    def __init__(self, path, use_openslide=False):
        self.path = path
        self.is_crashed = False
        self.crash_message = ""
        self.images = dict()
        try:
            if use_openslide:
                from openslide import OpenSlide
                slide = OpenSlide(path)
                self.level_tile_sizes = [
                        (256, 256) for _ in range(slide.level_count)]
                def read_tile(lv, row, col):
                    factor = int(self.slide.level_downsamples[lv])
                    loc = (256*row*factor, 256*col*factor)
                    size = (256, 256)
                    return self.slide.read_region(loc, lv, size)
                slide.read_tile = read_tile
            else:
                from eozinpy import Eozin
                slide = Eozin(path)
                self.level_tile_sizes = slide.level_tile_sizes
            lv_dims = slide.level_dimensions
            lowest_mg_lv = len(lv_dims) - 1
            lowest_mg_dim = lv_dims[-1]
            self.level_dimensions = lv_dims
            self.slide = slide
            self.lowest_mg_lv = lowest_mg_lv
            self.lowest_mg_dim = lowest_mg_dim
            img = slide.read_region((0, 0), lowest_mg_lv, lowest_mg_dim)
            self.lowest_mg_img = img
        except Exception as e:
            self.is_crashed = True
            self.crash_message = f"Eozin is crashed: {e}"

    def request(self, id_set):
        to_read = id_set - self.images.keys()
        res = dict()
        for ind in id_set:
            if ind in to_read:
                img = self.read_tile(ind)
                self.images[ind] = img
            else:
                img = self.images[ind]
            res[ind] = img
        return res

    def read_tile(self, ind):
        lv, row, col = ind
        img = self.slide.read_tile(lv, row, col)
        img = convert_pil_image_to_pyglet_image(img)
        return img


class TileGrid():
    """Helper class to calcurating which tile to be displayed
    """
    def __init__(
            self, 
            level_dimensions,
            level_tile_sizes,
            window_width,
            window_height,
            max_scale_form_low_mag=2.0):
        self.max_scale_lm = max_scale_form_low_mag
        self.lv_dims = level_dimensions
        self.lv_tile_sizes = level_tile_sizes
        self.lm_lev = len(level_dimensions) - 1 
        lm_width = level_dimensions[-1][0]
        self.mags = [s[0] / lm_width for s in level_dimensions]
        self.tile_ranges = [
            (self.diva(d[0], t[0]), self.diva(d[1], t[1]))
            for d, t in zip(level_dimensions, level_tile_sizes)
        ]
        self.w_width = window_width
        self.w_height = window_height

    def update_window_size(self, window_width, window_height):
        self.w_height = window_height
        self.w_width = window_width

    def require(self, sx, sy, lm_scale):
        """Calcurating tile index from current ROI
        """
        displayables = list(filter(
                lambda x: x >= lm_scale, 
                [self.max_scale_lm * m for m in self.mags]))
        if len(displayables) == 0:
            best_lv = 0
        else:
            best_lv = list(enumerate(displayables))[-1][0]
        if best_lv == self.lm_lev:
            return set()
        # Max id of row and col
        max_r, max_c = self.tile_ranges[best_lv]  
        img_scale = lm_scale / self.mags[best_lv]
        t_w, t_h = self.lv_tile_sizes[best_lv]
        st_w, st_h = t_w * img_scale, t_h * img_scale
        start_row = max(0, int(-1 * sx // st_w))
        start_col = max(0, int(-1 * sy // st_h))
        end_row = min(max_r, int((-1 * sx + self.w_width) // st_w) + 1)
        end_col = min(max_c, int((-1 * sy + self.w_height) // st_h) + 1)
        ts = {
                (best_lv, r, c) 
                for r in range(start_row, end_row) 
                for c in range(start_col, end_col)}
        return ts

    def draw_position(self, sx, sy, lm_scale, ind):
        """Calcurating tile position and scale from given index
        """
        lv, row, col = ind 
        img_scale = lm_scale / self.mags[lv]
        t_w, t_h = self.lv_tile_sizes[lv]
        st_w, st_h = t_w * img_scale, t_h * img_scale
        t_sx = sx + st_w * row
        t_sy = sy + st_h * col
        return t_sx, t_sy, img_scale

    @staticmethod
    def diva(num, den):
        if num % den == 0:
            return num // den
        else:
            return num // den + 1


class Coord():
    """Helper class for coordinate transformation
    
    Pyglet uses bottom-left is zero while Eozin, or 
    OpenSlide systems use top-left is zero slide-x
    """
    
    @staticmethod
    def slide2pyg(sx, sy, w_height, img_height=0):
        return sx, w_height - (sy + img_height)

    @staticmethod
    def pyg2slide(x, y, w_height, img_height=0):
        return x, w_height - (y + img_height)


def convert_pil_image_to_pyglet_image(pil_image):
    """Convert PIL type image to pyglet image

    :param PIL.Image.Image pil_image: PIL image
    :return: pyglet.image.ImageData
    """
    raw_img = pil_image.tobytes()
    pitch = -1 * pil_image.width * len(pil_image.mode)
    pyg_img = pyglet.image.ImageData(
        pil_image.width, pil_image.height, pil_image.mode,
        raw_img, pitch=pitch
    )
    return pyg_img


# App
def cli():
    parser = argparse.ArgumentParser(description="A simple WSI viewer")
    parser.add_argument("path", help="path of WSI file")
    parser.add_argument(
            "--openslide", 
            help="use OpenSlide backend",
            action="store_true")
    return parser.parse_args()


def app():
    args = cli()
    path = args.path
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    file_name = os.path.basename(path)
    reader = PathologyReader(path, use_openslide=args.openslide)
    w_height = 480
    window = pyglet.window.Window(
            width=640, 
            height=w_height, 
            resizable=True,
            caption=file_name)
    grid = TileGrid(
            level_dimensions=reader.level_dimensions,
            level_tile_sizes=reader.level_tile_sizes,
            window_width=window.width,
            window_height=window.height)

    model = Model(window_height=window.height)
    view = View(reader, window_height=w_height)
    ev = EventHandler(model=model, view=view, grid=grid)
    window.push_handlers(ev)

    @window.event
    def on_draw(*args, **kwargs):
        window.clear()
        view.on_draw()

    @window.event
    def on_close(*args, **kwargs):
        pyglet.app.exit()

    pyglet.app.run()


if __name__ == "__main__":
    app()

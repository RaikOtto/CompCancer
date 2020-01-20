import logging
import mahotas
import numpy as np
import scipy.ndimage
import scipy.special


logger = logging.getLogger(__name__)


def watershed(surface, markers, fg):
    # compute watershed
    ws = mahotas.cwatershed(surface, markers)

    # write watershed directly
    logger.debug("watershed output: %s %s %f %f",
                 ws.shape, ws.dtype, ws.max(), ws.min())

    # overlay fg and write
    wsFG = ws * fg
    logger.debug("watershed (foreground only): %s %s %f %f",
                 wsFG.shape, wsFG.dtype, wsFG.max(),
                 wsFG.min())
    wsFGUI = wsFG.astype(np.uint16)

    return wsFGUI


def label(prediction, kind, fg_thresh=0.9, seed_thresh=0.9):
    logger.info("labelling")

    if kind == "two_class":
        fg = 1.0 * (prediction[0] > fg_thresh)
        ws_surface = 1.0 - prediction[0]

        seeds = (1 * (prediction[0] > seed_thresh)).astype(np.uint8)
    elif kind == "affinities":
        # combine  components of affinities vector
        surface = 0.5 * (prediction[0] + prediction[1])
        # background pixel have affinity zero with everything
        # (including other bg pixel)
        fg = 1.0 * (surface > fg_thresh)
        ws_surface = 1.0 - surface

        seeds = (1 * (prediction > seed_thresh)).astype(np.uint8)
        seeds = (seeds[0] + seeds[1])
        seeds = (seeds > 1).astype(np.uint8)
    elif kind == 'three_class':
        # prediction[0] = bg
        # prediction[1] = inside
        # prediction[2] = boundary
        prediction = scipy.special.softmax(prediction, axis=0)
        fg = 1.0 * ((1.0 - prediction[0, ...]) > fg_thresh)
        ws_surface = 1.0 - prediction[1, ...]
        seeds = (1 * (prediction[1, ...] > seed_thresh)).astype(np.uint8)
    elif kind == 'sdt':
        # distance transform in negative inside an instance
        # so negative values correspond to fg
        if fg_thresh > 0:
            logger.warning("fg threshold should be zero/negative")
        fg = 1.0 * (prediction < fg_thresh)
        fg = fg.astype(np.uint8)

        ws_surface = prediction
        if seed_thresh > 0:
            logger.warning("surface/seed threshold should be negative")
        seeds = (1 * (ws_surface < seed_thresh)).astype(np.uint8)

    if np.count_nonzero(seeds) == 0:
        logger.warning("no seed points found for watershed")

    markers, cnt = scipy.ndimage.label(seeds)
    logger.info("num markers %s", cnt)

    # compute watershed
    labelling = watershed(ws_surface, markers, fg)

    return labelling, ws_surface

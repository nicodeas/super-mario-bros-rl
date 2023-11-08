from .bbox import BBox
import numpy as np
import cv2 as cv


# other constants (don't change these)
SCREEN_HEIGHT = 240
SCREEN_WIDTH = 256
MATCH_THRESHOLD = 0.9

MASK_COLOUR = np.array([252, 136, 104])

# LOAD IN TEMPLATES
# filenames for object templates
image_files = {
    "mario": {
        "small": [
            "./imaging/images/marioA.png",
            "./imaging/images/marioB.png",
            "./imaging/images/marioC.png",
            "./imaging/images/marioD.png",
            "./imaging/images/marioE.png",
            "./imaging/images/marioF.png",
            "./imaging/images/marioG.png",
        ],
        "tall": [
            "./imaging/images/tall_marioA.png",
            "./imaging/images/tall_marioB.png",
            "./imaging/images/tall_marioC.png",
        ],
        # Note: Many images are missing from tall mario, and I don't have any
        # images for fireball mario.
    },
    "enemy": {
        "goomba": ["./imaging/images/goomba.png"],
        "koopa": ["./imaging/images/koopaA.png", "./imaging/images/koopaB.png"],
    },
    "block": {
        "block": [
            "./imaging/images/block1.png",
            "./imaging/images/block2.png",
            "./imaging/images/block3.png",
            "./imaging/images/block4.png",
        ],
        "question_block": [
            "./imaging/images/questionA.png",
            "./imaging/images/questionB.png",
            "./imaging/images/questionC.png",
        ],
        "pipe": [
            "./imaging/images/pipe_upper_section.png",
            "./imaging/images/pipe_lower_section.png",
        ],
    },
    "item": {
        # Note: The template matcher is colourblind (it's using greyscale),
        # so it can't tell the difference between red and green mushrooms.
        "mushroom": ["./imaging/images/mushroom_red.png"],
        # There are also other items in the game that I haven't included,
        # such as star.
        # There's probably a way to change the matching to work with colour,
        # but that would slow things down considerably. Also, given that the
        # red and green mushroom sprites are so similar, it might think they're
        # the same even if there is colour.
    },
}


def _get_template(filename):
    image = cv.imread(filename)
    assert image is not None, f"File {filename} does not exist."
    template = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mask = np.uint8(np.where(np.all(image == MASK_COLOUR, axis=2), 0, 1))
    num_pixels = image.shape[0] * image.shape[1]
    if num_pixels - np.sum(mask) < 10:
        mask = None  # this is important for avoiding a problem where some things match everything
    dimensions = tuple(template.shape[::-1])
    return template, mask, dimensions


def get_template(filenames):
    results = []
    for filename in filenames:
        results.append(_get_template(filename))
    return results


def get_template_and_flipped(filenames):
    results = []
    for filename in filenames:
        template, mask, dimensions = _get_template(filename)
        results.append((template, mask, dimensions))
        results.append((cv.flip(template, 1), cv.flip(mask, 1), dimensions))
    return results


# Mario and enemies can face both right and left, so I'll also include
# horizontally flipped versions of those templates.
include_flipped = {"mario", "enemy"}


# generate all templatees
templates = {}
for category in image_files:
    category_items = image_files[category]
    category_templates = {}
    for object_name in category_items:
        filenames = category_items[object_name]
        if category in include_flipped or object_name in include_flipped:
            category_templates[object_name] = get_template_and_flipped(filenames)
        else:
            category_templates[object_name] = get_template(filenames)
    templates[category] = category_templates

################################################################################
# LOCATING OBJECTS


def _locate_object(screen, templates, stop_early=False, threshold=MATCH_THRESHOLD):
    locations = {}
    for template, mask, dimensions in templates:
        results = cv.matchTemplate(screen, template, cv.TM_CCOEFF_NORMED, mask=mask)
        locs = np.where(results >= threshold)
        for y, x in zip(*locs):
            locations[(x, y)] = dimensions

        # stop early if you found mario (don't need to look for other animation frames of mario)
        if stop_early and locations:
            break

    #      [((x,y), (width,height))]
    return [(loc, locations[loc]) for loc in locations]


def _locate_pipe(screen, threshold=MATCH_THRESHOLD) -> list[BBox]:
    upper_template, upper_mask, upper_dimensions = templates["block"]["pipe"][0]
    lower_template, lower_mask, lower_dimensions = templates["block"]["pipe"][1]

    # find the upper part of the pipe
    upper_results = cv.matchTemplate(
        screen, upper_template, cv.TM_CCOEFF_NORMED, mask=upper_mask
    )
    upper_locs = list(zip(*np.where(upper_results >= threshold)))

    # stop early if there are no pipes
    if not upper_locs:
        return []

    # find the lower part of the pipe
    lower_results = cv.matchTemplate(
        screen, lower_template, cv.TM_CCOEFF_NORMED, mask=lower_mask
    )
    lower_locs = set(zip(*np.where(lower_results >= threshold)))

    # put the pieces together
    upper_width, upper_height = upper_dimensions
    _, lower_height = lower_dimensions
    locations = []
    for y, x in upper_locs:
        for h in range(upper_height, SCREEN_HEIGHT, lower_height):
            if (y + h, x + 2) not in lower_locs:
                locations.append(BBox(x, y, upper_width, h, "pipe"))
                break
    return locations


def locate_objects(screen, mario_status) -> dict[str, list[BBox]]:
    # convert to greyscale
    screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)

    # iterate through our templates data structure
    object_locations = {}
    for category in templates:
        category_templates = templates[category]
        category_items = []
        stop_early = False
        for object_name in category_templates:
            # use mario_status to determine which type of mario to look for
            if category == "mario":
                if object_name != mario_status:
                    continue
                else:
                    stop_early = True
            # pipe has special logic, so skip it for now
            if object_name == "pipe":
                continue

            # find locations of objects
            results = _locate_object(
                screen, category_templates[object_name], stop_early
            )
            for location, dimensions in results:
                x, y = location
                width, height = dimensions
                category_items.append(BBox(x, y, width, height, object_name))

        object_locations[category] = category_items

    # locate pipes
    object_locations["block"] += _locate_pipe(screen)

    return object_locations

class BBox:
    #          y_min
    #           ___
    #          |   |  x_max
    #    x_min |   |
    #           ___
    #
    #          y_max
    #
    x_min: int
    x_max: int
    cx: int  # center x coordinate
    y_min: int
    y_max: int
    cy: int  # center y coordinate
    width: int
    height: int
    bbox_type: str

    def __init__(self, x, y, width, height, bbox_type) -> None:
        # you can get coordinates this way
        self.x_min = x
        self.x_max = x + width
        self.cx = x + (width / 2)
        self.y_min = y
        self.y_max = y + height
        self.cy = y + (height / 2)
        self.width = width
        self.height = height
        self.bbox_type = bbox_type

    def __repr__(self) -> str:
        return f"type:{self.bbox_type} x:{self.x_min} y:{self.y_min} width:{self.width} height:{self.height}"

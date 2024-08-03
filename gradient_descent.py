"""
https://en.wikipedia.org/wiki/gradient
https://en.wikipedia.org/wiki/Numerical_differentiation
https://en.wikipedia.org/wiki/gradient_descent
https://www.cs.cmu.edu/~ggordon/10725-F12/slides/05-gd-revisited.pdf
https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d
https://optimization.cbe.cornell.edu/index.php?title=Momentum
"""

from typing import Callable, Union
import time
import pygame
import math
import sys


WIDTH, HEIGHT = 950, 700
SECTION_SIZE = 20
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


pygame.init()
display = pygame.display
surface = display.set_mode((WIDTH, HEIGHT))
display.set_caption("Gradient Descent")


class Vector:
    def __init__(self, values: Union[list, tuple, set]) -> None:
        self.values = tuple([float(v) for v in values])

    def __add__(self, other: Union[int, float, "Vector"]) -> "Vector":
        if isinstance(other, (int, float)):
            return Vector(v + other for v in self.values)
        elif isinstance(other, Vector):
            assert len(self.values) == len(other), "invalid dimensions for addition"
            return Vector(self.values[i] + other[i] for i in range(len(other)))

    def __radd__(self, other: Union[int, float, "Vector"]) -> "Vector":
        return self.__add__(other)

    def __sub__(self, other: Union[int, float, "Vector"]) -> "Vector":
        if isinstance(other, (int, float)):
            return Vector(v - other for v in self.values)
        elif isinstance(other, Vector):
            assert len(self.values) == len(other), "invalid dimensions for subtraction"
            return Vector(self.values[i] - other[i] for i in range(len(other)))

    def __rsub__(self, other: Union[int, float, "Vector"]) -> "Vector":
        return self.__add__(other)

    def __mul__(self, other: Union[int, float, "Vector"]) -> "Vector":
        if isinstance(other, (int, float)):
            return Vector(v * other for v in self.values)
        elif isinstance(other, Vector):
            assert len(self.values) == len(other), "invalid dimensions for mul"
            return Vector(self.values[i] * other[i] for i in range(len(other)))

    def __rmul__(self, other: Union[int, float, "Vector"]) -> "Vector":
        return self.__mul__(other)

    def __pow__(self, power: Union[int, float]) -> "Vector":
        return Vector(v**power for v in self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> float:
        return self.values[index]

    def __str__(self) -> str:
        return " ".join([str(v) for v in self.values])

    def get_values(self) -> tuple:
        return self.values

    def magnitude(self) -> float:
        return math.sqrt(sum([v**2 for v in self.values]))


def function(p: Vector) -> float:
    return 300 * (
        0.5
        * math.exp(-((p[0] - 150) * (p[0] - 150) + (p[1] - 150) * (p[1] - 150)) / 10000)
        - 0.25
        * math.exp(-((p[0] - 200) * (p[0] - 200) + (p[1] - 400) * (p[1] - 400)) / 10000)
        - 0.37
        * math.exp(-((p[0] - 450) * (p[0] - 450) + (p[1] - 250) * (p[1] - 250)) / 8000)
        + 0.2
        * math.exp(-((p[0] - 700) * (p[0] - 700) + (p[1] - 550) * (p[1] - 550)) / 20000)
    )


def backtracking_line_search(
    f: Callable, p: Vector, grad: Vector, lr=0.70, max_step=100
) -> float:
    step = max_step
    while (f(p - step * grad)) > f(p) - (step / 2) * (grad.magnitude() ** 2):
        step *= lr

    return step


def gradient(f: Callable, p: Vector) -> Vector:
    grads = []
    for i in range(len(p)):
        h = abs(p[i]) * math.sqrt(2**-53) if p[i] != 0 else 1e-8
        g = (
            f(Vector([p[j] if j != i else p[j] + h for j in range(len(p))])) - f(p)
        ) / h
        grads.append(g)

    return Vector(grads)


def gradient_descent(
    p: Vector, error=1e-5, max_iters=100, v=0, momentum=0.7
) -> list[Vector]:
    positions = []
    i = 0
    while i < max_iters:
        grad = gradient(function, p)
        if all(abs(g) < error for g in grad):
            break
        step = backtracking_line_search(function, p, grad)
        v = momentum * v + (1 - momentum) * grad
        p = p - v * step
        positions.append(p)
        i += 1

    print(f"final point {p} found in {i} iterations")
    return positions


def quit_game() -> None:
    pygame.quit()
    sys.exit()


def calculate_color(value: float) -> tuple:
    if value > 0.5:
        r = int(max(0, min(255, value**3 * 255)))
    elif value < 0.5:
        r = 0.5 * int(max(0, min(255, (1 - value) ** 3 * 255)))
    else:
        r = 0
    g = int(max(0, min(255, value * 255)))
    b = int(max(0, min(255, (1 - value) * 255)))

    return (r, g, b)


def create_colormap() -> dict[tuple, tuple]:
    colormap = {}
    for x in range(0, WIDTH, SECTION_SIZE):
        for y in range(0, HEIGHT, SECTION_SIZE):
            section_sum = 0
            count = 0
            for i in range(SECTION_SIZE):
                for j in range(SECTION_SIZE):
                    if x + i < WIDTH and y + j < HEIGHT:
                        z = function((x + i, y + j))
                        section_sum += z
                        count += 1

            if count > 0:
                average_z = section_sum / count
                normalized_z = (average_z + 89.97000834699453) / (
                    149.95456143085883 + 89.97000834699453
                )
                color = calculate_color(normalized_z)
                colormap[(x, y)] = color

    return colormap


def main():
    p = Vector((-10, -10))

    run = True
    start = False

    clock = pygame.time.Clock()

    colormap = create_colormap()
    idx = 0
    while run:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                if p.get_values() == (-10, -10):
                    p = Vector(pos)
                    positions = gradient_descent(p)
                    positions.insert(0, p)
                    start = True

        surface.fill(BLACK)

        for x in range(0, WIDTH, SECTION_SIZE):
            for y in range(0, HEIGHT, SECTION_SIZE):
                pygame.draw.rect(
                    surface, colormap[(x, y)], (x, y, SECTION_SIZE, SECTION_SIZE)
                )

        pygame.draw.circle(surface, RED, p.get_values(), 7)
        if start:
            idx += 1
            if idx < len(positions):
                for i in range(1, idx + 1):
                    pygame.draw.circle(surface, BLUE, positions[i], 4)
                    pygame.draw.line(
                        surface,
                        RED,
                        positions[i - 1].get_values(),
                        positions[i].get_values(),
                        2,
                    )
            else:
                for i in range(1, len(positions)):
                    pygame.draw.circle(surface, BLUE, positions[i], 4)
                    pygame.draw.line(
                        surface,
                        RED,
                        positions[i - 1].get_values(),
                        positions[i].get_values(),
                        2,
                    )
                pygame.draw.circle(surface, GREEN, positions[-1], 4)

        pygame.display.flip()
        time.sleep(0.05)

    quit_game()


if __name__ == "__main__":
    main()

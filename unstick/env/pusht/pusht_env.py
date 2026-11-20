# PushT simulator adapted from Implicit Behavior Cloning (as distributed with the AdaJEPA code)
import gym
from gym import spaces
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
from typing import Tuple, Sequence
import pygame
import pymunk
import numpy as np
import shapely.geometry as sg
import cv2
import pymunk.pygame_util


positive_y_is_up: bool = False


class DrawOptions(pymunk.SpaceDebugDrawOptions):
    def __init__(self, surface: pygame.Surface) -> None:
        self.surface = surface
        super(DrawOptions, self).__init__()

    def draw_circle(
        self,
        pos: Vec2d,
        angle: float,
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p = to_pygame(pos, self.surface)

        pygame.draw.circle(self.surface, fill_color.as_int(), p, round(radius), 0)
        pygame.draw.circle(
            self.surface, light_color(fill_color).as_int(), p, round(radius - 4), 0
        )

    def draw_segment(self, a: Vec2d, b: Vec2d, color: SpaceDebugColor) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        pygame.draw.aalines(self.surface, color.as_int(), False, [p1, p2])

    def draw_fat_segment(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        r = round(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color.as_int(), False, [p1, p2], r)
        if r > 2:
            orthog = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
            if orthog[0] == 0 and orthog[1] == 0:
                return
            scale = radius / (orthog[0] * orthog[0] + orthog[1] * orthog[1]) ** 0.5
            orthog[0] = round(orthog[0] * scale)
            orthog[1] = round(orthog[1] * scale)
            points = [
                (p1[0] - orthog[0], p1[1] - orthog[1]),
                (p1[0] + orthog[0], p1[1] + orthog[1]),
                (p2[0] + orthog[0], p2[1] + orthog[1]),
                (p2[0] - orthog[0], p2[1] - orthog[1]),
            ]
            pygame.draw.polygon(self.surface, fill_color.as_int(), points)
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p1[0]), round(p1[1])),
                round(radius),
            )
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p2[0]), round(p2[1])),
                round(radius),
            )

    def draw_polygon(
        self,
        verts: Sequence[Tuple[float, float]],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        ps = [to_pygame(v, self.surface) for v in verts]
        ps += [ps[0]]

        radius = 2
        pygame.draw.polygon(self.surface, light_color(fill_color).as_int(), ps)

        if radius > 0:
            for i in range(len(verts)):
                a = verts[i]
                b = verts[(i + 1) % len(verts)]
                self.draw_fat_segment(a, b, radius, fill_color, fill_color)

    def draw_dot(
        self, size: float, pos: Tuple[float, float], color: SpaceDebugColor
    ) -> None:
        p = to_pygame(pos, self.surface)
        pygame.draw.circle(self.surface, color.as_int(), p, round(size), 0)


def to_pygame(p: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
    if positive_y_is_up:
        return round(p[0]), surface.get_height() - round(p[1])
    else:
        return round(p[0]), round(p[1])


def light_color(color: SpaceDebugColor):
    color = np.minimum(
        1.2 * np.float32([color.r, color.g, color.b, color.a]), np.float32([255])
    )
    color = SpaceDebugColor(r=color[0], g=color[1], b=color[2], a=color[3])
    return color


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f"Unsupported shape type {type(shape)}")
    geom = sg.MultiPolygon(geoms)
    return geom


class PushTEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0.0, 1.0)

    def __init__(
        self,
        render_size=224,
        reset_to_state=None,
        relative=True,
        action_scale=100,
        with_velocity=False,
        with_target=True,
        shape="T",  # shape can be "T" <- the original shape, "I", "L", "Z", "square" and "small_tee"
        color="LightSlateGray",
        agent_color="RoyalBlue",
        goal_color=None,  # None => default ("LightGreen" with target, else "White")
    ):
        self.shape = shape
        self.color = color
        self.agent_color = agent_color
        self.goal_color_override = goal_color
        self._seed = None
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20  # PD control.z
        self.control_hz = self.metadata["video.frames_per_second"]
        self.relative = relative  # relative action space
        self.action_scale = action_scale

        # agent_pos, block_pos, block_angle
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float64),
            high=np.array([ws, ws, ws, ws, np.pi * 2], dtype=np.float64),
            shape=(5,),
            dtype=np.float64,
        )

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float64),
            high=np.array([ws, ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64,
        )

        self.space = None

        self.with_velocity = with_velocity
        self.with_target = with_target
        self.reset_to_state = reset_to_state
        self.coverage_arr = []

    def reset(self):
        self._setup()
        state = self.reset_to_state  # set by prepare()
        self._set_state(state)

        self.coverage_arr = []
        state = self._get_obs()
        visual = self._render_frame("rgb_array")
        proprio = state[:2]
        if self.with_velocity:
            proprio = np.concatenate((proprio, state[5:]))
        observation = {
            "visual": visual,
            "proprio": proprio
        }
        return observation, state

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            action = np.array(action) * self.action_scale
            if self.relative:
                action = self.agent.position + action
            for i in range(n_steps):
                # Step PD control.
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (
                    Vec2d(0, 0) - self.agent.velocity
                )
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)

        # compute reward
        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        reward = np.clip(coverage / self.success_threshold, 0, 1)
        done = False  # coverage > self.success_threshold

        self.coverage_arr.append(coverage)

        state = self._get_obs()
        visual = self._render_frame("rgb_array")
        proprio = state[:2]
        if self.with_velocity:
            proprio = np.concatenate((proprio, state[5:]))
        observation = {
            "visual": visual,
            "proprio": proprio
        }
        info = self._get_info()
        info["state"] = state
        info["max_coverage"] = max(self.coverage_arr)
        info["final_coverage"] = self.coverage_arr[-1]

        return observation, reward, done, info

    def _get_obs(self):
        if self.with_velocity:
            obs = np.array(
                tuple(self.agent.position)
                + tuple(self.block.position)
                + (self.block.angle % (2 * np.pi),)
                + tuple(self.agent.velocity)
            ).astype(np.float32)
        else:
            obs = np.array(
                tuple(self.agent.position)
                + tuple(self.block.position)
                + (self.block.angle % (2 * np.pi),)
            ).astype(np.float32)
        return obs

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # legacy assignment order (kept for compatibility)
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            "pos_agent": np.array(self.agent.position),
            "vel_agent": np.array(self.agent.velocity),
            "block_pose": np.array(list(self.block.position) + [self.block.angle]),
            "goal_pose": self.goal_pose,
            "n_contacts": n_contact_points_per_step,
        }
        return info

    def _render_frame(self, mode):
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [
                pymunk.pygame_util.to_pygame(
                    goal_body.local_to_world(v), draw_options.surface
                )
                for v in shape.get_vertices()
            ]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        img = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        img = cv2.resize(img, (self.render_size, self.render_size))
        return img

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        vel_block = tuple(state[5:]) if self.with_velocity else (0, 0)
        self.agent.velocity = vel_block
        self.agent.position = pos_agent
        # setting angle rotates with respect to center of mass, so set it first
        self.block.angle = rot_block
        self.block.position = pos_block

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0

        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agent = self.add_circle((256, 400), 15)
        # self.block = self.add_tee((256, 300), 0)
        self.block = self.add_shape(self.shape, (256, 300), 0, color=self.color, scale=40)
        if getattr(self, "goal_color_override", None) is not None:
            self.goal_color = pygame.Color(self.goal_color_override)
        elif self.with_target:
            self.goal_color = pygame.Color("LightGreen")
        else:
            self.goal_color = pygame.Color("White")
        self.goal_pose = np.array([256, 256, np.pi / 4])  # x, y, theta (in radians)

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.95  # 95% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color(
            "LightGray"
        )  # standard color names
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color(getattr(self, "agent_color", "RoyalBlue"))
        self.space.add(body, shape)
        return body

    def add_tee(
        self,
        position,
        angle,
        scale=30,
        color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):  
        scale=30
        mass = 1
        length = 4
        vertices1 = [
            (-length * scale / 2, scale),
            (length * scale / 2, scale),
            (length * scale / 2, 0),
            (-length * scale / 2, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (
            shape1.center_of_gravity + shape2.center_of_gravity
        ) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body
    
    def add_small_tee(
        self,
        position,
        angle,
        scale=30,
        color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        vertices1 = [
            (-3 * scale / 2, scale),
            (3 * scale / 2, scale),
            (3 * scale / 2, 0),
            (-3 * scale / 2, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, 2 * scale),
            (scale / 2, 2 * scale),
            (scale / 2, scale),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (
            shape1.center_of_gravity + shape2.center_of_gravity
        ) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body

    def add_plus(
        self,
        position,
        angle,
        scale=30,
        color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        vertices1 = [
            (-3 * scale / 2, scale / 2),
            (3 * scale / 2, scale / 2),
            (3 * scale / 2, -scale / 2),
            (-3 * scale / 2, -scale / 2),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (-scale / 2, scale / 2),
            (-scale / 2, 3 * scale / 2),
            (scale / 2, scale / 2),
            (scale / 2, 3 * scale / 2),
        ]
        vertices3 = [
            (-scale / 2, -scale / 2),
            (-scale / 2, -3 * scale / 2),
            (scale / 2, -scale / 2),
            (scale / 2, -3 * scale / 2),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        inertia3 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2 + inertia3)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape3 = pymunk.Poly(body, vertices3)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape3.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        shape3.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (
            shape1.center_of_gravity + shape2.center_of_gravity + shape3.center_of_gravity
        ) / 3
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2, shape3)
        return body

    def add_L(
        self,
        position,
        angle,
        scale=30,
        color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        length = 2
        vertices1 = [
            (0, 0),
            (0, scale * length),
            (scale * length / 2, scale * length),
            (scale * length / 2, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (0, 0),
            (scale * length, 0),
            (scale * length, -scale * length / 2),
            (0, -scale * length / 2),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (
            shape1.center_of_gravity + shape2.center_of_gravity
        ) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body

    def add_Z(
        self,
        position,
        angle,
        scale=30,
        color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        length = 2
        vertices1 = [
            (0, 0),
            (0, length * scale / 2),
            (length * scale, length * scale / 2),
            (length * scale, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (-length * scale / 2, 0),
            (length * scale / 2, 0),
            (length * scale / 2, -length * scale / 2),
            (-length * scale / 2, -length * scale / 2),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (
            shape1.center_of_gravity + shape2.center_of_gravity
        ) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body

    def add_square(
        self,
        position,
        angle,
        scale=30,
        color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        vertices1 = [
            (-scale, -scale),
            (-scale, scale),
            (scale, scale),
            (scale, -scale),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1)
        shape1 = pymunk.Poly(body, vertices1)
        shape1.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = shape1.center_of_gravity
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1)
        return body

    def add_I(
        self,
        position,
        angle,
        scale=30,
        color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        vertices1 = [
            (-scale / 2, -scale * 2),
            (-scale / 2, scale * 2),
            (scale / 2, scale * 2),
            (scale / 2, -scale * 2),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1)
        shape1 = pymunk.Poly(body, vertices1)
        shape1.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = shape1.center_of_gravity
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1)
        return body

    def add_shape(self, shape, *args, **kwargs):
        # Dispatch method based on the 'shape' parameter
        if shape == "L":
            return self.add_L(*args, **kwargs)
        elif shape == "T":
            return self.add_tee(*args, **kwargs)
        elif shape == "Z":
            return self.add_Z(*args, **kwargs)
        elif shape == "square":
            return self.add_square(*args, **kwargs)
        elif shape == "I":
            return self.add_I(*args, **kwargs)
        elif shape == "small_tee":
            return self.add_small_tee(*args, **kwargs)
        if shape == "+":
            return self.add_plus(*args, **kwargs)
        else:
            raise ValueError(f"Unknown shape type: {shape}")

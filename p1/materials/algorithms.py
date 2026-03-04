import random
from collections import deque  # for bfs
from heapq import heappush, heappop  # for branch and bound

import itertools

import config
from sprites import Spaceship, Goal


class Algorithm:
    def get_path(self, state):
        pass


class ExampleAlgorithm(Algorithm):
    def get_path(self, state):
        path = []
        while not state.is_goal_state():
            possible_actions = state.get_legal_actions()
            action = possible_actions[random.randint(0, len(possible_actions) - 1)]
            path.append(action)
            state = state.generate_successor_state(action)
        return path


class DFS(Algorithm):
    def get_path(self, state):
        stack = [(state, [])]
        visited = set()

        while stack:
            current_state, path = stack.pop()

            if current_state.is_goal_state():
                return path

            visited.add(current_state)

            possible_actions = current_state.get_legal_actions()

            for action in reversed(possible_actions):
                successor_state = current_state.generate_successor_state(action)

                if successor_state not in visited:
                    stack.append((successor_state, path + [action]))

        return []


class BFS(Algorithm):
    def get_path(self, state):
        queue = deque([(state, [])])
        visited = set()

        while queue:
            current_state, path = queue.popleft()

            if current_state.is_goal_state():
                return path

            visited.add(current_state)

            possible_actions = current_state.get_legal_actions()

            for action in possible_actions:
                successor_state = current_state.generate_successor_state(action)

                if successor_state not in visited:
                    queue.append((successor_state, path + [action]))

        return []


class BNB(Algorithm):
    def get_path(self, state):
        queue = []
        counter = itertools.count()
        min_costs = {}

        # (cost_so_far, action_priority, unique_id, current_state, path_to_state)
        heappush(queue, (0, 0, next(counter), state, []))
        min_costs[state] = 0

        while queue:
            cost_so_far, _, _, current_state, path = heappop(queue)

            if current_state.is_goal_state():
                return path

            possible_actions = current_state.get_legal_actions()

            for priority, action in enumerate(possible_actions):
                successor_state = current_state.generate_successor_state(action)
                action_cost = state.get_action_cost(action)
                new_cost = cost_so_far + action_cost

                if successor_state not in min_costs or new_cost < min_costs[successor_state]:
                    min_costs[successor_state] = new_cost
                    heappush(queue, (new_cost, priority, next(counter), successor_state, path + [action]))

        return []


class AStar(Algorithm):
    def __init__(self):
        super().__init__()
        self.goal_assignments = {}

    @staticmethod
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    @staticmethod
    def get_positions_from_bitmap(bitmap):
        positions = []
        for index in range(bitmap.bit_length()):
            if (bitmap >> index) & 1:
                row = index // config.N
                col = index % config.N
                positions.append((row, col))
        return positions

    def assign_goals(self, state):
        ships_bitmap = state.get_state(Spaceship.kind())
        goals_bitmap = state.get_state(Goal.kind())

        ships = AStar.get_positions_from_bitmap(ships_bitmap)
        goals = AStar.get_positions_from_bitmap(goals_bitmap)

        unassigned_goals = set(goals)
        assignments = {}

        for ship in ships:
            closest_goal = min(unassigned_goals, key=lambda goal: self.manhattan_distance(ship, goal))
            assignments[ship] = closest_goal
            unassigned_goals.remove(closest_goal)

        self.goal_assignments = assignments

    def heuristic(self, state):
        ships_bitmap = state.get_state(Spaceship.kind())
        ships = AStar.get_positions_from_bitmap(ships_bitmap)

        return sum(self.manhattan_distance(ship, self.goal_assignments[ship]) for ship in ships)

    def get_path(self, state):
        self.assign_goals(state)

        queue = []
        counter = itertools.count()
        min_costs_without_heuristic = {}

        initial_cost = 0

        heuristic_cost = self.heuristic(state)
        heappush(queue, (initial_cost + heuristic_cost, 0, next(counter), state, []))
        min_costs_without_heuristic[state] = initial_cost

        while queue:
            cost_so_far_with_heuristic, _, _, current_state, path = heappop(queue)
            cost_so_far_without_heuristic = min_costs_without_heuristic[current_state]

            if current_state.is_goal_state():
                return path

            possible_actions = current_state.get_legal_actions()

            for priority, action in enumerate(possible_actions):
                successor_state = current_state.generate_successor_state(action)
                action_cost = state.get_action_cost(action)
                new_cost_without_heuristic = cost_so_far_without_heuristic + action_cost

                if (successor_state not in min_costs_without_heuristic or
                        new_cost_without_heuristic < min_costs_without_heuristic[successor_state]):
                    min_costs_without_heuristic[successor_state] = new_cost_without_heuristic

                    self.assign_goals(successor_state)
                    heuristic_cost = self.heuristic(successor_state)
                    heappush(queue,
                             (new_cost_without_heuristic + heuristic_cost,
                              priority,
                              next(counter),
                              successor_state,
                              path + [action])
                             )

        return []


class Greedy(Algorithm):
    def __init__(self):
        super().__init__()
        self.goal_assignments = {}

    @staticmethod
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    @staticmethod
    def get_positions_from_bitmap(bitmap):
        positions = []
        for index in range(bitmap.bit_length()):
            if (bitmap >> index) & 1:
                row = index // config.N
                col = index % config.N
                positions.append((row, col))
        return positions

    def assign_goals(self, state):
        ships_bitmap = state.get_state(Spaceship.kind())
        goals_bitmap = state.get_state(Goal.kind())

        ships = AStar.get_positions_from_bitmap(ships_bitmap)
        goals = AStar.get_positions_from_bitmap(goals_bitmap)

        unassigned_goals = set(goals)
        assignments = {}

        for ship in ships:
            closest_goal = min(unassigned_goals, key=lambda goal: self.manhattan_distance(ship, goal))
            assignments[ship] = closest_goal
            unassigned_goals.remove(closest_goal)

        self.goal_assignments = assignments

    def heuristic(self, state):
        ships_bitmap = state.get_state(Spaceship.kind())
        ships = AStar.get_positions_from_bitmap(ships_bitmap)

        return sum(self.manhattan_distance(ship, self.goal_assignments[ship]) for ship in ships)

    def get_path(self, state):
        self.assign_goals(state)

        queue = []
        counter = itertools.count()
        visited = set()

        heuristic_cost = self.heuristic(state)
        heappush(queue, (heuristic_cost, 0, next(counter), state, []))

        while queue:
            heuristic, _, _, current_state, path = heappop(queue)

            if current_state.is_goal_state():
                return path

            visited.add(current_state)

            possible_actions = current_state.get_legal_actions()

            for priority, action in enumerate(possible_actions):
                successor_state = current_state.generate_successor_state(action)

                if successor_state not in visited:

                    self.assign_goals(successor_state)
                    heuristic_cost = self.heuristic(successor_state)
                    heappush(queue,
                             (heuristic_cost,
                              priority,
                              next(counter),
                              successor_state,
                              path + [action])
                             )

        return []


class IDAStar(Algorithm):
    def __init__(self):
        super().__init__()
        self.goal_assignments = {}

    @staticmethod
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    @staticmethod
    def get_positions_from_bitmap(bitmap):
        positions = []
        for index in range(bitmap.bit_length()):
            if (bitmap >> index) & 1:
                row = index // config.N
                col = index % config.N
                positions.append((row, col))
        return positions

    def assign_goals(self, state):
        ships_bitmap = state.get_state(Spaceship.kind())
        goals_bitmap = state.get_state(Goal.kind())

        ships = AStar.get_positions_from_bitmap(ships_bitmap)
        goals = AStar.get_positions_from_bitmap(goals_bitmap)

        unassigned_goals = set(goals)
        assignments = {}

        for ship in ships:
            closest_goal = min(unassigned_goals, key=lambda goal: self.manhattan_distance(ship, goal))
            assignments[ship] = closest_goal
            unassigned_goals.remove(closest_goal)

        self.goal_assignments = assignments

    def heuristic(self, state):
        ships_bitmap = state.get_state(Spaceship.kind())
        ships = AStar.get_positions_from_bitmap(ships_bitmap)

        return sum(self.manhattan_distance(ship, self.goal_assignments[ship]) for ship in ships)

    def search(self, state, threshold):
        self.assign_goals(state)

        queue = []
        counter = itertools.count()
        min_costs_without_heuristic = {}

        initial_cost = 0

        heuristic_cost = self.heuristic(state)
        heappush(queue, (initial_cost + heuristic_cost, 0, next(counter), state, []))
        min_costs_without_heuristic[state] = initial_cost

        while queue:
            cost_so_far_with_heuristic, _, _, current_state, path = heappop(queue)
            cost_so_far_without_heuristic = min_costs_without_heuristic[current_state]

            if current_state.is_goal_state():
                return path
            elif cost_so_far_with_heuristic > threshold:
                return cost_so_far_with_heuristic

            possible_actions = current_state.get_legal_actions()

            for priority, action in enumerate(possible_actions):
                successor_state = current_state.generate_successor_state(action)
                action_cost = state.get_action_cost(action)
                new_cost_without_heuristic = cost_so_far_without_heuristic + action_cost

                if (successor_state not in min_costs_without_heuristic or
                        new_cost_without_heuristic < min_costs_without_heuristic[successor_state]):
                    min_costs_without_heuristic[successor_state] = new_cost_without_heuristic

                    self.assign_goals(successor_state)
                    heuristic_cost = self.heuristic(successor_state)
                    heappush(queue,
                             (new_cost_without_heuristic + heuristic_cost,
                              priority,
                              next(counter),
                              successor_state,
                              path + [action])
                             )

        return []

    def get_path(self, state):
        self.assign_goals(state)

        heuristic_cost = self.heuristic(state)
        threshold = heuristic_cost
        print("Initial threshold: ", threshold)

        while True:
            result = self.search(state, threshold)

            if isinstance(result, list):
                return result

            threshold = result
            print("New threshold: ", threshold)

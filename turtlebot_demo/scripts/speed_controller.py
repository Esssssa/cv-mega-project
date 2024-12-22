class SpeedController:
    def __init__(self, base_linear_speed, min_linear_speed, angular_speed, slowdown_margin):
        self.base_linear_speed = base_linear_speed
        self.min_linear_speed = min_linear_speed
        self.angular_speed = angular_speed
        self.slowdown_margin = slowdown_margin

    def adjust_linear_speed(self, line_center, frame_width):
        if line_center is None:
            return 0
        
        edge_margin = int(self.slowdown_margin * frame_width)
        if line_center < edge_margin or line_center > frame_width - edge_margin:
            return self.min_linear_speed
        
        distance_from_center = abs(line_center - frame_width / 2)
        max_distance = frame_width / 2 - edge_margin
        speed_factor = 1 - (distance_from_center / max_distance)
        
        return self.min_linear_speed + (self.base_linear_speed - self.min_linear_speed) * speed_factor

    def calculate_angular_velocity(self, line_center, frame_width):
        if line_center is None:
            return 0
        
        error = line_center - (frame_width / 2)
        angular_vel = -error * 0.005
        return max(min(angular_vel, self.angular_speed), -self.angular_speed)
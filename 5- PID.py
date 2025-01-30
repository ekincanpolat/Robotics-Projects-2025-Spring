Kp = 5.0  # Proportional gain
Ki = 0.1  # Integral gain
Kd = 0.2  # Derivative gain

#initial
class PIDController:
    def __init__(self, Kp, Ki, Kd, dt=1/240):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.prev_error = 0
        self.integral = 0

#PID controller
    def compute(self, target, current):
        error = target - current
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative
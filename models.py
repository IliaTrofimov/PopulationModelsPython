import numpy as np
from scipy.integrate import solve_ivp


class BaseOdeModel:
    def __call__(self, t: float, yy: np.ndarray | list[float] | tuple[float]):
            raise NotImplemented()

    def solve(self, x0: float, y0: float, t: float, dt: float = 0.1):
        return solve_ivp(self.__call__, (0, t), (x0, y0), first_step=dt).y

    def validate_params(self):
        pass

    def latex_str(self):
        return "$x' = f(t, x, y)$\n$y' = g(t, x, y)$"

    def enum_parameters(self):
        return {}

    def set_param(self, p_name: str, value):
        pass


class BazykinModelA(BaseOdeModel):
    def __init__(self, alpha: float = 1, gamma: float = 1, eps: float = 1, mu: float = 1):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.mu = mu

    def __call__(self, t: float, yy: np.ndarray | list[float] | tuple[float]):
        return (yy[0] - yy[0]*yy[1] / (1 + self.alpha*yy[0]) - self.eps*yy[0]**2,
                -self.gamma*yy[1] + yy[0]*yy[1] / (1 + self.alpha*yy[0]) - self.mu*yy[1]**2)

    def validate_params(self):
        if self.alpha < 0: self.alpha = 0.0
        elif self.alpha > 10: self.alpha = 10.0

        if self.gamma < 0: self.gamma = 0.0
        elif self.gamma > 10: self.gamma = 10.0

        if self.eps < 0: self.eps = 0.0
        elif self.eps > 10: self.eps = 10.0

        if self.mu < 0: self.mu = 0.0
        elif self.mu > 10: self.mu = 10.0

    def __str__(self):
        return (f"x' = x - xy/(1 + {self.eps:.3f}x) - {self.eps:.3f}x^2\n"
                f"y' = {-self.gamma:.3f}y + xy/(1 + {self.eps:.3f}x) - {self.mu:.3f}y^2")

    def latex_str(self):
        return (f"$x' = x - \\frac{{xy}}{{1 + {self.eps:.3f}x}} - {self.eps:.3f}x^2$\n"
                f"$y' = {-self.gamma:.3f}y + \\frac{{xy}}{{1 + {self.eps:.3f}x}} - {self.mu:.3f}y^2$")

    def enum_parameters(self):
        return {
            "α": (self.alpha, 0, 10),
            "γ": (self.gamma, 0, 10),
            "μ": (self.mu, 0, 10),
            "ε": (self.eps, 0, 10)
        }

    def set_param(self, p_name: str, value: float):
        match p_name:
            case "α":
                self.alpha = value
            case "γ":
                self.gamma = value
            case "μ":
                self.mu = value
            case "ε":
                self.eps = value
            case _:
                raise KeyError(f"Unknown model parameter '{p_name}'")

    def get_param(self, p_name: str):
        match p_name:
            case "α": return self.alpha
            case "γ": return self.gamma
            case "μ": return self.mu
            case "ε": return self.eps
            case _:
                raise KeyError(f"Unknown model parameter '{p_name}'")

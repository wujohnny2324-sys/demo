"""
Mathematical Formulas Library
Comprehensive collection of mathematical, statistical, and financial formulas.
"""

import math
from typing import List, Tuple, Union, Callable


# ============================================================================
# ARITHMETIC OPERATIONS
# ============================================================================

def add(a: float, b: float) -> float:
    """Addition: a + b"""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtraction: a - b"""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiplication: a * b"""
    return a * b


def divide(a: float, b: float) -> float:
    """Division: a / b"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def modulo(a: float, b: float) -> float:
    """Modulo: a % b"""
    if b == 0:
        raise ValueError("Cannot modulo by zero")
    return a % b


def power(base: float, exponent: float) -> float:
    """Power: base^exponent"""
    return base ** exponent


def sqrt(n: float) -> float:
    """Square root: √n"""
    if n < 0:
        raise ValueError("Cannot take square root of negative number")
    return math.sqrt(n)


def cbrt(n: float) -> float:
    """Cube root: ∛n"""
    if n < 0:
        return -abs(n) ** (1/3)
    return n ** (1/3)


def nth_root(n: float, root: float) -> float:
    """nth root: n^(1/root)"""
    if n < 0 and root % 2 == 0:
        raise ValueError("Cannot take even root of negative number")
    return n ** (1 / root)


# ============================================================================
# TRIGONOMETRIC FUNCTIONS
# ============================================================================

def sin(angle_rad: float) -> float:
    """Sine function (radians)"""
    return math.sin(angle_rad)


def cos(angle_rad: float) -> float:
    """Cosine function (radians)"""
    return math.cos(angle_rad)


def tan(angle_rad: float) -> float:
    """Tangent function (radians)"""
    return math.tan(angle_rad)


def cot(angle_rad: float) -> float:
    """Cotangent: 1/tan(x)"""
    sin_val = math.sin(angle_rad)
    if sin_val == 0:
        raise ValueError("Cotangent undefined")
    return math.cos(angle_rad) / sin_val


def sec(angle_rad: float) -> float:
    """Secant: 1/cos(x)"""
    cos_val = math.cos(angle_rad)
    if cos_val == 0:
        raise ValueError("Secant undefined")
    return 1 / cos_val


def csc(angle_rad: float) -> float:
    """Cosecant: 1/sin(x)"""
    sin_val = math.sin(angle_rad)
    if sin_val == 0:
        raise ValueError("Cosecant undefined")
    return 1 / sin_val


def asin(x: float) -> float:
    """Arcsine (inverse sine)"""
    if not -1 <= x <= 1:
        raise ValueError("Arcsine input must be in [-1, 1]")
    return math.asin(x)


def acos(x: float) -> float:
    """Arccosine (inverse cosine)"""
    if not -1 <= x <= 1:
        raise ValueError("Arccosine input must be in [-1, 1]")
    return math.acos(x)


def atan(x: float) -> float:
    """Arctangent (inverse tangent)"""
    return math.atan(x)


def atan2(y: float, x: float) -> float:
    """Two-argument arctangent"""
    return math.atan2(y, x)


# ============================================================================
# EXPONENTIAL & LOGARITHMIC FUNCTIONS
# ============================================================================

def exp(x: float) -> float:
    """Exponential: e^x"""
    return math.exp(x)


def ln(x: float) -> float:
    """Natural logarithm: ln(x)"""
    if x <= 0:
        raise ValueError("Logarithm input must be positive")
    return math.log(x)


def log10(x: float) -> float:
    """Base 10 logarithm: log₁₀(x)"""
    if x <= 0:
        raise ValueError("Logarithm input must be positive")
    return math.log10(x)


def log(x: float, base: float) -> float:
    """Logarithm with custom base: logₐ(x)"""
    if x <= 0 or base <= 0 or base == 1:
        raise ValueError("Invalid logarithm parameters")
    return math.log(x, base)


# ============================================================================
# STATISTICAL FUNCTIONS
# ============================================================================

def mean(data: List[float]) -> float:
    """Arithmetic mean (average)"""
    if not data:
        raise ValueError("Data list cannot be empty")
    return sum(data) / len(data)


def median(data: List[float]) -> float:
    """Median (middle value)"""
    if not data:
        raise ValueError("Data list cannot be empty")
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 1:
        return sorted_data[n // 2]
    return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2


def mode(data: List[float]) -> float:
    """Mode (most frequent value)"""
    if not data:
        raise ValueError("Data list cannot be empty")
    return max(set(data), key=data.count)


def variance(data: List[float], sample: bool = False) -> float:
    """Variance (population or sample)"""
    if not data:
        raise ValueError("Data list cannot be empty")
    avg = mean(data)
    var = sum((x - avg) ** 2 for x in data)
    return var / (len(data) - 1 if sample else len(data))


def stdev(data: List[float], sample: bool = False) -> float:
    """Standard deviation (population or sample)"""
    return math.sqrt(variance(data, sample))


def zscore(x: float, mean_val: float, stdev_val: float) -> float:
    """Z-score: (x - μ) / σ"""
    if stdev_val == 0:
        raise ValueError("Standard deviation cannot be zero")
    return (x - mean_val) / stdev_val


def covariance(x: List[float], y: List[float]) -> float:
    """Covariance between two datasets"""
    if len(x) != len(y) or not x:
        raise ValueError("Lists must have equal non-zero length")
    mean_x = mean(x)
    mean_y = mean(y)
    return sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))) / len(x)


def correlation(x: List[float], y: List[float]) -> float:
    """Pearson correlation coefficient"""
    if len(x) != len(y) or not x:
        raise ValueError("Lists must have equal non-zero length")
    cov = covariance(x, y)
    std_x = stdev(x)
    std_y = stdev(y)
    if std_x == 0 or std_y == 0:
        raise ValueError("Standard deviation cannot be zero")
    return cov / (std_x * std_y)


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile (0-100)"""
    if not 0 <= p <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    if not data:
        raise ValueError("Data list cannot be empty")
    sorted_data = sorted(data)
    index = (p / 100) * (len(sorted_data) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_data) - 1)
    weight = index - lower
    return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight


# ============================================================================
# GEOMETRIC FUNCTIONS
# ============================================================================

def circle_area(radius: float) -> float:
    """Area of circle: πr²"""
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    return math.pi * radius ** 2


def circle_perimeter(radius: float) -> float:
    """Circumference of circle: 2πr"""
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    return 2 * math.pi * radius


def sphere_volume(radius: float) -> float:
    """Volume of sphere: (4/3)πr³"""
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    return (4/3) * math.pi * radius ** 3


def sphere_surface_area(radius: float) -> float:
    """Surface area of sphere: 4πr²"""
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    return 4 * math.pi * radius ** 2


def rectangle_area(length: float, width: float) -> float:
    """Area of rectangle: l × w"""
    if length < 0 or width < 0:
        raise ValueError("Dimensions must be non-negative")
    return length * width


def rectangle_perimeter(length: float, width: float) -> float:
    """Perimeter of rectangle: 2(l + w)"""
    if length < 0 or width < 0:
        raise ValueError("Dimensions must be non-negative")
    return 2 * (length + width)


def triangle_area(base: float, height: float) -> float:
    """Area of triangle: (1/2)bh"""
    if base < 0 or height < 0:
        raise ValueError("Dimensions must be non-negative")
    return 0.5 * base * height


def triangle_area_heron(a: float, b: float, c: float) -> float:
    """Area using Heron's formula: √[s(s-a)(s-b)(s-c)]"""
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError("Side lengths must be positive")
    s = (a + b + c) / 2
    area_sq = s * (s - a) * (s - b) * (s - c)
    if area_sq < 0:
        raise ValueError("Invalid triangle sides")
    return math.sqrt(area_sq)


def cylinder_volume(radius: float, height: float) -> float:
    """Volume of cylinder: πr²h"""
    if radius < 0 or height < 0:
        raise ValueError("Dimensions must be non-negative")
    return math.pi * radius ** 2 * height


def cylinder_surface_area(radius: float, height: float) -> float:
    """Surface area of cylinder: 2πr(r + h)"""
    if radius < 0 or height < 0:
        raise ValueError("Dimensions must be non-negative")
    return 2 * math.pi * radius * (radius + height)


def cone_volume(radius: float, height: float) -> float:
    """Volume of cone: (1/3)πr²h"""
    if radius < 0 or height < 0:
        raise ValueError("Dimensions must be non-negative")
    return (1/3) * math.pi * radius ** 2 * height


def cone_surface_area(radius: float, slant_height: float) -> float:
    """Surface area of cone: πr(r + l)"""
    if radius < 0 or slant_height < 0:
        raise ValueError("Dimensions must be non-negative")
    return math.pi * radius * (radius + slant_height)


def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance in 2D"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def distance_3d(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
    """Euclidean distance in 3D"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


# ============================================================================
# COMBINATORIAL FUNCTIONS
# ============================================================================

def factorial(n: int) -> int:
    """Factorial: n!"""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    return math.factorial(n)


def permutations(n: int, r: int) -> int:
    """Permutations: P(n,r) = n!/(n-r)!"""
    if n < 0 or r < 0 or r > n:
        raise ValueError("Invalid permutation parameters")
    return factorial(n) // factorial(n - r)


def combinations(n: int, r: int) -> int:
    """Combinations: C(n,r) = n!/(r!(n-r)!)"""
    if n < 0 or r < 0 or r > n:
        raise ValueError("Invalid combination parameters")
    return factorial(n) // (factorial(r) * factorial(n - r))


# ============================================================================
# ALGEBRAIC FUNCTIONS
# ============================================================================

def quadratic(a: float, b: float, c: float, x: float) -> float:
    """Evaluate quadratic: ax² + bx + c"""
    return a * x ** 2 + b * x + c


def quadratic_roots(a: float, b: float, c: float) -> Tuple[float, float]:
    """Quadratic formula: (-b ± √(b²-4ac)) / 2a"""
    if a == 0:
        raise ValueError("Coefficient 'a' cannot be zero")
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        raise ValueError("No real solutions (negative discriminant)")
    sqrt_disc = math.sqrt(discriminant)
    x1 = (-b + sqrt_disc) / (2 * a)
    x2 = (-b - sqrt_disc) / (2 * a)
    return (x1, x2)


def linear(m: float, x: float, b: float) -> float:
    """Linear equation: y = mx + b"""
    return m * x + b


def absolute(x: float) -> float:
    """Absolute value: |x|"""
    return abs(x)


# ============================================================================
# NUMBER THEORY
# ============================================================================

def gcd(a: int, b: int) -> int:
    """Greatest common divisor"""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """Least common multiple"""
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


def is_prime(n: int) -> bool:
    """Check if number is prime"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def fibonacci(n: int) -> int:
    """Fibonacci sequence: F(n)"""
    if n < 0:
        raise ValueError("Fibonacci index must be non-negative")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


# ============================================================================
# FINANCIAL FORMULAS
# ============================================================================

def simple_interest(principal: float, rate: float, time: float) -> float:
    """Simple interest: I = P × r × t"""
    if principal < 0 or rate < 0 or time < 0:
        raise ValueError("Parameters must be non-negative")
    return principal * rate * time


def compound_interest(principal: float, rate: float, time: float, n: int = 1) -> float:
    """Compound interest: A = P(1 + r/n)^(nt)"""
    if principal < 0 or rate < 0 or time < 0 or n < 1:
        raise ValueError("Invalid parameters")
    return principal * (1 + rate / n) ** (n * time)


def continuous_compound(principal: float, rate: float, time: float) -> float:
    """Continuous compound interest: A = Pe^(rt)"""
    if principal < 0 or rate < 0 or time < 0:
        raise ValueError("Invalid parameters")
    return principal * math.exp(rate * time)


def present_value(future_value: float, rate: float, time: float) -> float:
    """Present value: PV = FV / (1 + r)^t"""
    if rate < 0 or time < 0:
        raise ValueError("Invalid parameters")
    return future_value / (1 + rate) ** time


def future_value(present_value: float, rate: float, time: float) -> float:
    """Future value: FV = PV(1 + r)^t"""
    if rate < 0 or time < 0:
        raise ValueError("Invalid parameters")
    return present_value * (1 + rate) ** time


def npv(discount_rate: float, cash_flows: List[float]) -> float:
    """Net present value"""
    if discount_rate < 0:
        raise ValueError("Discount rate cannot be negative")
    return sum(cf / (1 + discount_rate) ** i for i, cf in enumerate(cash_flows))


# ============================================================================
# CALCULUS APPROXIMATIONS
# ============================================================================

def numerical_derivative(f: Callable, x: float, h: float = 1e-5) -> float:
    """Numerical derivative using central difference"""
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_integral_trapezoid(f: Callable, a: float, b: float, n: int = 1000) -> float:
    """Numerical integration using trapezoidal rule"""
    if a >= b or n < 1:
        raise ValueError("Invalid parameters")
    h = (b - a) / n
    result = (f(a) + f(b)) / 2
    for i in range(1, n):
        result += f(a + i * h)
    return result * h


if __name__ == "__main__":
    print("Mathematical Formulas Library")
    print("=" * 60)
    print(f"add(10, 5) = {add(10, 5)}")
    print(f"multiply(6, 7) = {multiply(6, 7)}")
    print(f"sqrt(144) = {sqrt(144)}")
    print(f"sin(π/2) = {sin(math.pi / 2)}")
    print(f"ln(e) = {ln(math.e)}")
    print(f"mean([1, 2, 3, 4, 5]) = {mean([1, 2, 3, 4, 5])}")
    print(f"circle_area(5) = {circle_area(5):.2f}")
    print(f"factorial(6) = {factorial(6)}")
    print(f"fibonacci(10) = {fibonacci(10)}")
    print(f"is_prime(23) = {is_prime(23)}")
    print(f"compound_interest(1000, 0.05, 2) = {compound_interest(1000, 0.05, 2):.2f}")
    print(f"quadratic_roots(1, -5, 6) = {quadratic_roots(1, -5, 6)}")

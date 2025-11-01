import math

def binomial_pmf(x: int, n: int, p: float) -> float:
    """
    Berechnet P(X = x) für eine Binomialverteilung B(n,p)
    x : Anzahl der Treffer
    n : Anzahl der Versuche
    p : Trefferwahrscheinlichkeit pro Versuch
    """
    # Binomialkoeffizient: n über x
    binom = math.comb(n, x)
    return binom * (p**x) * ((1-p)**(n-x))

def binomial_expectation(n: int, p: float) -> float:
    """
    Erwartungswert E[X] einer Binomialverteilung B(n,p)
    n : Anzahl der Versuche
    p : Trefferwahrscheinlichkeit pro Versuch
    """
    return n * p

def binomial_variance(n: int, p: float) -> float:
    "Varianz[x] einer Binomialverteilung B(n,p)"
    return n * p * (1 - p)

def binomial_standard_deviation(n: int, p: float) -> float:
    "Standardabweichung einer Binomialverteilung B(n,p)"
    return math.sqrt(binomial_variance(n, p))
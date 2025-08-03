import sympy as sp
import threading
import logging

logger = logging.getLogger(__name__)

class SymbolRegistry:
    """Thread-safe singleton symbol registry"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize symbols once"""
        try:
            # Real symbols
            self.x = sp.Symbol('x', real=True)
            self.alpha = sp.Symbol('alpha', real=True)
            self.beta = sp.Symbol('beta', real=True)
            self.M = sp.Symbol('M', real=True)
            self.z = sp.Symbol('z', real=True)
            self.t = sp.Symbol('t', real=True)
            self.s = sp.Symbol('s', real=True)
            self.omega = sp.Symbol('omega', real=True)
            
            # Positive symbols
            self.a = sp.Symbol('a', positive=True)
            self.q = sp.Symbol('q', positive=True)
            self.v = sp.Symbol('v', positive=True)
            self.n = sp.Symbol('n', positive=True)
            
            # Function symbols
            self.y = sp.Function('y')
            
            # Mathematical constants
            self.pi = sp.pi
            self.e = sp.E
            self.I = sp.I  # Imaginary unit
            
            self._initialized = True
            logger.debug("Symbol registry initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing symbol registry: {e}")
            raise
    
    def __getstate__(self):
        """Support pickling"""
        return self.__dict__
    
    def __setstate__(self, state):
        """Support unpickling"""
        self.__dict__.update(state)
    
    def get_all_symbols(self):
        """Return dictionary of all symbols"""
        return {
            'x': self.x,
            'alpha': self.alpha,
            'beta': self.beta,
            'M': self.M,
            'z': self.z,
            't': self.t,
            's': self.s,
            'omega': self.omega,
            'a': self.a,
            'q': self.q,
            'v': self.v,
            'n': self.n,
            'y': self.y,
            'pi': self.pi,
            'e': self.e,
            'I': self.I
        }

# Global singleton instance
SYMBOLS = SymbolRegistry()

# For backward compatibility and convenience
x = SYMBOLS.x
alpha = SYMBOLS.alpha
beta = SYMBOLS.beta
M = SYMBOLS.M
z = SYMBOLS.z
t = SYMBOLS.t
s = SYMBOLS.s
omega = SYMBOLS.omega
a = SYMBOLS.a
q = SYMBOLS.q
v = SYMBOLS.v
n = SYMBOLS.n
y = SYMBOLS.y
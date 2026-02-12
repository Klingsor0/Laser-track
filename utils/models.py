import numpy as np
class modelo_lineal:
    """
    clase para calculo de parametros de modelo lineal atraves de mínimos cuadrados
    """

    def __init__(self, x:list[float], y:list[float], intercepcion:bool = True) -> None:
        if len(x) != len(y):
            raise ValueError("Las listas de datos 'x' e 'y' deben tener la misma longitud")
        self.x = x
        self.y = y
        self.n = len(x)
        self._media = []
        self._coeficiente = None
        self._rss = None
        self._se = []
        self._inter = intercepcion
        self._intercepcion = None if intercepcion is not None else 0.0

    @property
    def media(self) -> float:
        """
        media
        """
        if not self._media:
            self._media.append(sum(self.x) / self.n)
            self._media.append(sum(self.y) / self.n)
        return self._media

    @property
    def coeficiente(self) -> float:
        """
        coeficiente de pendiente de modelo lineal
        """
        if not self._coeficiente:
            if self._inter:
                coef1 = 0
                coef2 = 0
                # coef = sum((xi - self.media[0])*yi for xi, yi in zip(self.x, self.y)) / sum((xi - self.media[0])**2 for xi in self.x ) if self_inter else sum(s)
                for i in range(self.n):
                      term = (self.x[i] - self.media[0])
                      term1 = term ** 2 
                      term *= self.y[i]
                      coef1 += term
                      coef2 += term1
                self._coeficiente = coef1 / coef2
            else: 
               self._coeficiente = sum(xi*yi for xi, yi in zip(self.x,self.y)) / sum(xi**2 for xi in self.x) 
        return self._coeficiente
    
    @property
    def intercepcion(self) -> float:
        """
        valor de intercepción
        """
        if not self._intercepcion:
            self._intercepcion = self.media[1] - self.coeficiente * self.media[0] if self._inter else 0.0
        return self._intercepcion

    @property
    def rss(self) -> float:
        """
        suma de residuos cuadraticos
        """
        if self._rss is None:
            self._rss = sum((yi - pred)**2 for yi, pred in zip(self.y, self.predictions))
        return self._rss

    @property
    def predictions(self) -> list[float]:
        """
        Retorna la lista de valores estimados (predichos) para cada x.
        """
        return [self.coeficiente * xi + self.intercepcion for xi in self.x] 

    @property
    def R2(self) -> float:
        """
        coeficiente de determinación
        """
        tss = sum((yi - self.media[1])**2 for yi in self.y) if self._inter else sum(yi**2 for yi in self.y)
        return 1 - self.rss / tss if tss != 0 else 0.0

    @property
    def se(self) -> tuple[float]:
        """
        error estandar
        """
        denom = sum( (xi - self.media[0])**2 for xi in self.x )
        num = sum( xi**2 for xi in self.x )

        if self._inter:
            self._se.append(sqrt((num/(denom*self.n))*self.rss/ (self.n - 2) ))
            self._se.append(sqrt((1/(self.n - 2))* self.rss / denom ))
        else:
            self._se.append(sqrt((self.rss/(self.n - 1))* (1/ num)))
        return self._se
    
    def resumen(self) -> None:
        """
        Impresión de resumen en tabla de los coeficientes y estadisticos
        """
        print(f"Coeficiente de determinación R²: {self.R2:.4f}")
        formato_renglon = "{:>15}" * 3
        print(formato_renglon.format("", "Valor calculado", "Error Estándar"))

        if self._inter:
            print(formato_renglon.format("Intercepción", f"{self.intercepcion: .4f}", f"{self.se[0]: .4f}") )
            print(formato_renglon.format("Pendiente", f"{self.coeficiente: .4f}", f"{self.se[1]: .4f}") )
        else: 
            print(formato_renglon.format("Pendiente", f"{self.coeficiente: .4f}", f"{self.se[0]: .4f}") )


class modelo_parabolico:
    """
    clase para calculo de parametros de modelo parabolico por inversión matricial. 
    La formula se ha obtenido del libro de Bertha Oda, Introducción al análisis grafico de datos experimentales
    """

    def __init__(self, x:list[float], y:list[float]) -> None:
        if len(x) != len(y):
            raise ValueError("Las listas de datos 'x' e 'y' deben tener la misma longitud")
        self.x = x
        self.y = y
        self.n = len(x)
        self._coeficientes = None
        self._matriz = None
        self._rss = None
        self._se = []
        self.calculo_matriz()
        self.calculo_coeficientes()

    def calculo_matriz(self) -> None:
        """
        calcula las componentes de la matriz
        """
        if not self._matriz:
            A = np.zeros((3,3))
            A[0,0] = self.n
            A[0,1] = sum(self.x)
            A[1,0] = A[0,1]
            A[0,2] = sum([xi**2 for xi in self.x])
            A[2,0] = A[0,2]
            A[1,1] = A[0,2]
            A[1,2] = sum([xi**3 for xi in self.x])
            A[2,1] = A[1,2]
            A[2,2] = sum([xi**4 for xi in self.x])
            self._matriz = A
    
    def calculo_coeficientes(self) -> None:
        """
        calculo de coeficientes por medio de inversión de matriz
        """
        if not self._coeficientes:
            lhs = np.array([sum(self.y), 
                            sum([xi*yi for xi, yi in zip(self.x, self.y)]), 
                            sum([yi*xi**2 for xi, yi in zip(self.x, self.y)])])
            matriz_inv = np.linalg.inv(self._matriz)
            self._coeficientes = matriz_inv @ lhs

    def predict(self, new_x:list[float]) -> list[float]:
        """
        Retorna la lista de valores estimados (predichos) para cada x.
        """
        return [self._coeficientes[0]  + self._coeficientes[1]*xi + self._coeficientes[2] * xi**2 for xi in new_x] 


    




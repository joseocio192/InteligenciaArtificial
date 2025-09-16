public class Arbol {
    Nodo raiz;

    public Arbol() {
        this.raiz = null;
    }

    public void insertar(int valor) {
        this.raiz = insertarRecursivo(this.raiz, valor);
    }

    private Nodo insertarRecursivo(Nodo nodo, int valor) {
        if (vacio(nodo)) {
            return new Nodo(valor);
        }
        if (valor < nodo.getValor()) {
            nodo.setIzquierdo(insertarRecursivo(nodo.getIzquierdo(), valor));
        } else if (valor > nodo.getValor()) {
            nodo.setDerecho(insertarRecursivo(nodo.getDerecho(), valor));
        }
        return nodo;
    }

    public void ImprimirArbol() {
        ImprimirArbolRecursivo(this.raiz);
    }

    private void ImprimirArbolRecursivo(Nodo nodo) {
        if (nodo != null) {
            ImprimirArbolRecursivo(nodo.getIzquierdo());
            System.out.print(nodo.getValor() + " ");
            ImprimirArbolRecursivo(nodo.getDerecho());
        }
    }

    public Nodo buscarNodo(int valor) {
        return buscarNodoRecursivo(this.raiz, valor);
    }

    private Nodo buscarNodoRecursivo(Nodo nodo, int valor) {
        if (vacio(nodo) || nodo.getValor() == valor) {
            return nodo;
        }
        if (valor < nodo.getValor()) {
            return buscarNodoRecursivo(nodo.getIzquierdo(), valor);
        } else {
            return buscarNodoRecursivo(nodo.getDerecho(), valor);
        }
    }

    public boolean vacio(Nodo nodo) {
        return nodo == null;
    }
}

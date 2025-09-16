public class Nodo {
    int valor;
    Nodo izquierdo;
    Nodo derecho;

    public Nodo(int valor) {
        this.valor = valor;
        this.izquierdo = null;
        this.derecho = null;
    }

    public int getValor() {
        return valor;
    }

    public void setValor(int valor) {
        this.valor = valor;
    }

    public Nodo getIzquierdo() {
        return izquierdo;
    }

    public void setIzquierdo(Nodo izquierdo) {
        this.izquierdo = izquierdo;
    }

    public Nodo getDerecho() {
        return derecho;
    }

    public void setDerecho(Nodo derecho) {
        this.derecho = derecho;
    }
}

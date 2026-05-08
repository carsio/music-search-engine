import { Button } from "../../primitives/Button";
import styles from "./ErrorState.module.css";

interface ErrorStateProps {
  message?: string;
  onRetry?: () => void;
}

export function ErrorState({ message, onRetry }: ErrorStateProps) {
  return (
    <div className={styles.box} role="alert">
      <div className={styles.iconBox} aria-hidden>!</div>
      <h2 className={styles.title}>Algo deu errado.</h2>
      {message ? <p className={styles.message}>{message}</p> : null}
      {onRetry ? (
        <Button variant="outline" onClick={onRetry}>
          Tentar de novo
        </Button>
      ) : null}
    </div>
  );
}

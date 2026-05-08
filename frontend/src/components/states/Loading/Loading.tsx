import { cx } from "../../../utils/classnames";
import styles from "./Loading.module.css";

interface LoadingProps {
  variant?: "panel" | "list" | "home" | "inline";
  label?: string;
}

export function Loading({ variant = "inline", label }: LoadingProps) {
  if (variant === "inline") {
    return (
      <div className={styles.inline} role="status">
        <span className={styles.spinner} aria-hidden />
        {label ? <span>{label}</span> : null}
      </div>
    );
  }
  if (variant === "list") {
    return (
      <div className={styles.list} aria-busy="true">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className={cx(styles.skeleton, styles.skeletonRow)} />
        ))}
      </div>
    );
  }
  if (variant === "home") {
    return (
      <div className={styles.home} aria-busy="true">
        <div className={cx(styles.skeleton, styles.skeletonHero)} />
      </div>
    );
  }
  return (
    <div className={styles.panel} aria-busy="true">
      <div className={cx(styles.skeleton, styles.skeletonBlock)} />
      <div className={cx(styles.skeleton, styles.skeletonText)} />
      <div className={cx(styles.skeleton, styles.skeletonText)} />
      <div className={cx(styles.skeleton, styles.skeletonTextShort)} />
    </div>
  );
}

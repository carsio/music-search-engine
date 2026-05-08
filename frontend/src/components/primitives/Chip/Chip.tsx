import type { ButtonHTMLAttributes, ReactNode } from "react";
import { cx } from "../../../utils/classnames";
import styles from "./Chip.module.css";

interface ChipProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  active?: boolean;
  variant?: "default" | "quick" | "related";
  eyebrow?: string;
  sub?: string;
  children: ReactNode;
}

export function Chip({
  active,
  variant = "default",
  eyebrow,
  sub,
  children,
  className,
  ...rest
}: ChipProps) {
  if (variant === "related") {
    return (
      <button
        className={cx(styles.related, active && styles.activeRelated, className)}
        {...rest}
      >
        {eyebrow ? <span className={styles.eyebrow}>{eyebrow}</span> : null}
        <span className={styles.title}>{children}</span>
        {sub ? <span className={styles.sub}>{sub}</span> : null}
      </button>
    );
  }
  return (
    <button
      className={cx(
        styles.chip,
        variant === "quick" && styles.quick,
        active && styles.active,
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}

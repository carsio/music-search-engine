import type { ButtonHTMLAttributes, ReactNode } from "react";
import { cx } from "../../../utils/classnames";
import styles from "./Button.module.css";

type Variant = "primary" | "ghost" | "outline" | "cta";
type Size = "sm" | "md" | "lg";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  icon?: ReactNode;
  children?: ReactNode;
}

export function Button({
  variant = "primary",
  size = "md",
  icon,
  children,
  className,
  ...rest
}: ButtonProps) {
  return (
    <button
      className={cx(styles.btn, styles[variant], styles[size], className)}
      {...rest}
    >
      {icon ? <span className={styles.icon}>{icon}</span> : null}
      {children ? <span className={styles.label}>{children}</span> : null}
    </button>
  );
}

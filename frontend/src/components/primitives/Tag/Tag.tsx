import type { ReactNode } from "react";
import { cx } from "../../../utils/classnames";
import styles from "./Tag.module.css";

interface TagProps {
  children: ReactNode;
  variant?: "pill" | "ri" | "score";
  onClick?: () => void;
  className?: string;
}

export function Tag({ children, variant = "pill", onClick, className }: TagProps) {
  const Component = onClick ? "button" : "span";
  return (
    <Component
      className={cx(styles.tag, styles[variant], className)}
      onClick={onClick}
      type={onClick ? "button" : undefined}
    >
      {children}
    </Component>
  );
}

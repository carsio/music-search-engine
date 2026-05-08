import type { HTMLAttributes, ReactNode } from "react";
import { cx } from "../../../utils/classnames";
import styles from "./Card.module.css";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  variant?: "flat" | "raised" | "muted" | "dashed";
  padding?: "none" | "sm" | "md" | "lg";
  children: ReactNode;
}

export function Card({
  variant = "flat",
  padding = "md",
  children,
  className,
  ...rest
}: CardProps) {
  return (
    <div
      className={cx(
        styles.card,
        styles[variant],
        styles[`p-${padding}`],
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
}

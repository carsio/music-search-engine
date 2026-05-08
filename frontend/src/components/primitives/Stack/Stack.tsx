import type { CSSProperties, ElementType, HTMLAttributes, ReactNode } from "react";
import { cx } from "../../../utils/classnames";
import styles from "./Stack.module.css";

interface StackProps extends HTMLAttributes<HTMLDivElement> {
  as?: ElementType;
  direction?: "row" | "column";
  gap?: number | string;
  align?: CSSProperties["alignItems"];
  justify?: CSSProperties["justifyContent"];
  wrap?: boolean;
  children: ReactNode;
}

export function Stack({
  as: Component = "div",
  direction = "column",
  gap = 12,
  align,
  justify,
  wrap,
  children,
  className,
  style,
  ...rest
}: StackProps) {
  return (
    <Component
      className={cx(styles.stack, className)}
      style={{
        flexDirection: direction,
        gap: typeof gap === "number" ? `${gap}px` : gap,
        alignItems: align,
        justifyContent: justify,
        flexWrap: wrap ? "wrap" : undefined,
        ...style,
      }}
      {...rest}
    >
      {children}
    </Component>
  );
}

import type { ReactNode } from "react";
import { cx } from "../../../utils/classnames";
import styles from "./ResultsLayout.module.css";

interface ResultsLayoutProps {
  primary: ReactNode;
  secondary?: ReactNode;
  variant?: "two-col" | "one-col";
  eyebrow?: ReactNode;
}

export function ResultsLayout({
  primary,
  secondary,
  variant = "two-col",
  eyebrow,
}: ResultsLayoutProps) {
  const useTwoCol = variant === "two-col" && !!secondary;
  return (
    <div className={cx(styles.layout, useTwoCol ? styles.twoCol : styles.oneCol)}>
      <div className={styles.main}>
        {eyebrow ? <div className={styles.eyebrow}>{eyebrow}</div> : null}
        {primary}
      </div>
      {useTwoCol ? <aside className={styles.side}>{secondary}</aside> : null}
    </div>
  );
}

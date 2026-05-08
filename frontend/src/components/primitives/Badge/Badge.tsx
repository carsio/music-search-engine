import type { ReactNode } from "react";
import { cx } from "../../../utils/classnames";
import styles from "./Badge.module.css";

type Tone = "artist" | "track" | "lyric" | "genre" | "album" | "neutral";

interface BadgeProps {
  tone?: Tone;
  children: ReactNode;
  className?: string;
}

export function Badge({ tone = "neutral", children, className }: BadgeProps) {
  return (
    <span className={cx(styles.badge, styles[tone], className)}>{children}</span>
  );
}

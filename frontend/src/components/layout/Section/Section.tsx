import type { ReactNode } from "react";
import styles from "./Section.module.css";

interface SectionProps {
  title?: string;
  meta?: ReactNode;
  children: ReactNode;
}

export function Section({ title, meta, children }: SectionProps) {
  return (
    <section className={styles.section}>
      {title || meta ? (
        <div className={styles.head}>
          {title ? <h2 className={styles.title}>{title}</h2> : <span />}
          {meta ? <div className={styles.meta}>{meta}</div> : null}
        </div>
      ) : null}
      {children}
    </section>
  );
}

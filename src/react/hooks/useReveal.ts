import { useEffect, useRef, useState } from "react";

/**
 * Sets `active` when the element intersects the viewport (once), for scroll-driven “lift” animations.
 */
export function useReveal(rootMargin = "0px 0px -8% 0px") {
  const ref = useRef<HTMLElement | null>(null);
  const [active, setActive] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            setActive(true);
            break;
          }
        }
      },
      { threshold: 0.1, rootMargin },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [rootMargin]);

  return { ref, active };
}

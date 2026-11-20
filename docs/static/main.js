// Project page behaviour: theme, nav state, reveal, comparison slider, tabs, BibTeX copy, lightbox.
(function () {
  var root = document.documentElement;
  root.classList.add("js");

  // theme: saved choice, else the system setting
  var saved = null;
  try { saved = localStorage.getItem("theme"); } catch (e) {}
  if (saved) root.setAttribute("data-theme", saved);
  else if (window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches) root.setAttribute("data-theme", "dark");
  document.addEventListener("DOMContentLoaded", function () {
    var tb = document.querySelector(".theme-btn");
    if (tb) tb.addEventListener("click", function () {
      var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) {}
    });

    // nav border and active link
    var nav = document.querySelector(".nav");
    var links = Array.prototype.slice.call(document.querySelectorAll(".nav-links a"));
    var secs = links.map(function (a) { return document.querySelector(a.getAttribute("href")); });
    function onScroll() {
      if (nav) nav.classList.toggle("scrolled", window.scrollY > 8);
      var cur = -1;
      secs.forEach(function (s, i) { if (s && s.getBoundingClientRect().top < 120) cur = i; });
      links.forEach(function (a, i) { a.classList.toggle("active", i === cur); });
    }
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();

    // reveal on scroll
    var els = document.querySelectorAll(".reveal");
    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (es) {
        es.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
      }, { rootMargin: "0px 0px -8% 0px" });
      els.forEach(function (el) { io.observe(el); });
    } else els.forEach(function (el) { el.classList.add("in"); });

    // comparison slider
    document.querySelectorAll(".compare").forEach(function (c) {
      var r = c.querySelector("input[type=range]");
      function set() { c.style.setProperty("--pos", r.value + "%"); }
      r.addEventListener("input", set);
      set();
    });

    // tabs
    document.querySelectorAll("[role=tablist]").forEach(function (tl) {
      var tabs = tl.querySelectorAll("[role=tab]");
      tabs.forEach(function (t) {
        t.addEventListener("click", function () {
          tabs.forEach(function (o) {
            var on = o === t;
            o.setAttribute("aria-selected", on ? "true" : "false");
            document.getElementById(o.getAttribute("aria-controls")).hidden = !on;
          });
        });
      });
    });

    // BibTeX copy
    document.querySelectorAll(".copy").forEach(function (b) {
      b.addEventListener("click", function () {
        var txt = document.querySelector(b.getAttribute("data-target")).innerText;
        var done = function () { var l = b.querySelector("span"); var o = l.textContent; l.textContent = "Copied"; setTimeout(function () { l.textContent = o; }, 1600); };
        if (navigator.clipboard) navigator.clipboard.writeText(txt).then(done);
        else { var ta = document.createElement("textarea"); ta.value = txt; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove(); done(); }
      });
    });

    // lightbox
    var lb = document.createElement("div");
    lb.className = "lightbox";
    lb.setAttribute("role", "dialog");
    lb.setAttribute("aria-label", "Enlarged figure");
    lb.innerHTML = "<img alt=''>";
    document.body.appendChild(lb);
    var big = lb.querySelector("img");
    document.querySelectorAll("img[data-zoom]").forEach(function (im) {
      im.addEventListener("click", function () { big.src = im.currentSrc || im.src; big.alt = im.alt; lb.classList.add("open"); });
    });
    lb.addEventListener("click", function () { lb.classList.remove("open"); });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") lb.classList.remove("open"); });
  });
})();

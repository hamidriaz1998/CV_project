// ─────────────────────────────────────────────
//  Lab Cover Page Template
//  Usage: #import "lab-cover-template.typ": lab-cover
// ─────────────────────────────────────────────

#let lab-cover(
  lab-title:    "Lab Title",
  logo:         none,
  session:      "Session XXXX – XXXX",
  student-name: "Student Name",
  student-id:   "XXXX-XX-XX",
  supervisor:   "Supervisor Name",
  course:       "Course Code: Course Name",
  department:   "Department Name",
  university:   "University Name",
  location:     "City, Country",
) = {
  set page(
    width: 210mm,
    height: 297mm,
    margin: 2cm,
  )
  set text(
    font: "Times New Roman",
    size: 12pt,
  )

  align(center, {
    // ── Lab Title ──────────────────────────────
    text(size: 22pt, weight: "bold", lab-title)
    v(0.5cm)

    // ── Logo (optional) ────────────────────────
    if logo != none {
      image(logo, width: 4cm)
    }
    v(1cm)

    // ── Session ────────────────────────────────
    text(size: 16pt, weight: "bold", session)
    v(1.08cm)

    // ── Submitted by ───────────────────────────
    text(size: 16pt, weight: "bold", "Submitted by:")
    v(0.5cm)
    text(size: 16pt, student-name + h(1cm) + student-id)
    v(1.08cm)

    // ── Supervised by ──────────────────────────
    text(size: 16pt, weight: "bold", "Supervised by:")
    v(0.5cm)
    text(size: 16pt, supervisor)
    v(1.08cm)

    // ── Course ─────────────────────────────────
    text(size: 16pt, weight: "bold", "Course:")
    v(0.5cm)
    text(size: 16pt, course)
    v(1.08cm)

    // ── Department & University ────────────────
    text(size: 16pt, department)
    v(0.5cm)
    text(size: 18pt, weight: "bold", university)
    linebreak()
    text(size: 18pt, weight: "bold", location)
  })
}

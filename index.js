const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const stepTime = 0.1;
const markInterval = 10;

// Physics
const DENSITY = 7850; // kg/m³ (steel)
const GRAVITY = 1;

// Dots
const r = 0.025;
const offset = 0.4;
const v0 = 1;

class Sphere {
  constructor(p = 7850, r = 0.01, x = 0, y = 0, vx = 0, vy = 0, rigid = false){
    this.p = p; // kg/m³ (default steel)
    this.r = r; // m
    this.x = x; // m
    this.y = y; // m
    this.vx = vx; // m/s
    this.vy = vy; // m/s
    this.ax = 0; // m/s^2
    this.ay = 0; // m/s^2
    this.m = this.calcMass(p, r) // kg
    this.rigid = rigid;
  }

  calcMass(p, r){
    return 4 / 3 * Math.PI * Math.pow(r, 3) * p;
  }

  draw() {
    //draw a circle
    const scale = 1000;

    ctx.beginPath();
    ctx.arc(this.x * scale, this.y * scale, this.r * scale, 0, Math.PI*2, true);
    ctx.closePath();
    ctx.fill();
  }
}

function moveWithGravity(dt, o) {  // "o" refers to Array of objects we are moving
  for (let o1 of o) {  // Zero-out accumulator of forces for each object
    if(o1.rigid) break;
    o1.fx = 0;
    o1.fy = 0;
  }
  for (let [i, o1] of o.entries()) {  // For each pair of objects...
    if(o1.rigid) break;
    for (let [j, o2] of o.entries()) {
      if (i < j) {  // To not do same pair twice
        let dx = o2.x - o1.x;  // Compute distance between centers of objects
        let dy = o2.y - o1.y;
        let r = Math.sqrt(Math.pow(dx, 2) + Math.pow(dy, 2));
        if (r < 1) {  // To avoid division by 0
            r = 1;
        }
        // Compute force for this pair; k = 1000
        let f = (GRAVITY * o1.m * o2.m) / Math.pow(r, 2);
        let fx = f * dx / r;  // Break it down into components
        let fy = f * dy / r;
        o1.fx += fx;  // Accumulate for first object
        o1.fy += fy;
        o2.fx -= fx;  // And for second object in opposite direction
        o2.fy -= fy;
      }
    }
  }
  for (let o1 of o) {  // for each object update...
    if(o1.rigid) break;
    let ax = o1.fx / o1.m;  // ...acceleration
    let ay = o1.fy / o1.m;

    o1.vx += ax * dt;  // ...speed
    o1.vy += ay * dt;

    o1.x += o1.vx * dt;  // ...position
    o1.y += o1.vy * dt;
  }
}

const dots = [
  new Sphere(DENSITY, r, 0.050 + offset, 0.050 + offset, v0, 0),
  new Sphere(DENSITY, r, 0.450 + offset, 0.050 + offset, 0, v0),
  new Sphere(DENSITY, r, 0.050 + offset, 0.450 + offset, 0, -v0),
  new Sphere(DENSITY, r, 0.450 + offset, 0.450 + offset, -v0, 0),
  new Sphere(DENSITY, r, 0.250 + offset, 0.250 + offset, 0, 0),
  new Sphere(DENSITY, r, 0.450 + offset, 0.450 + offset, -v0, 0),
  new Sphere(DENSITY, r, 0.250 + offset, 0.250 + offset, 0, 0)
];

// Track mouse's position
let mouseX = 0;
let mouseY = 0;

window.addEventListener('mousemove', (event) => {
	mouseX = event.pageX - canvas.offsetLeft;
	mouseY = event.pageY - canvas.offsetTop;
});

let timerStamp = new Date().getTime();

function animate() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);


  moveWithGravity(stepTime, dots);

  //console.log(dots);

  const now = new Date().getTime();
  if(now - timerStamp > markInterval){
    timerStamp = now;
    for (let dot of dots) {
      if(!dot.rigid){
        const mark = new Sphere(DENSITY, dot.r/10, dot.x, dot.y);
        mark.rigid = true;
        dots.push(mark);
      }
    }
  }

  for (let dot of dots) {
    dot.draw();
  }

  window.requestAnimationFrame(animate);
}

window.requestAnimationFrame(animate);
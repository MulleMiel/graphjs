let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");


let offsetAngle = 0;
let lastTime = new Date().getTime();

const speed = 1; // deg/s


function animate() {
  const time = new Date().getTime();
  const dt = (time - lastTime) / 1000;
  lastTime = time;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.beginPath();

  ctx.arc(100, 100, 100, 0 + 2 * Math.sin(offsetAngle / 2), Math.PI + 2 * Math.sin(offsetAngle));
  //ctx.fill();
  ctx.stroke();

  offsetAngle += speed * 2 * Math.PI * dt;

  console.log(offsetAngle);

  requestAnimationFrame(animate);
}

animate();
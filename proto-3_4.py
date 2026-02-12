import streamlit as st
# import importlib
import cv2 as cv
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pandas as pd
# import ast
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from utils import snake1 as snk
from utils.models import modelo_parabolico
from scipy.interpolate import UnivariateSpline

#importlib.reload(utils)
#importlib.reload(utils.snake)
# CARGA DE IMAGEN
# hay dos opciones o cargar la imagen en openCV y transformarla a objeto pil o al revés.

@st.cache_data
def load_img(uploaded):
    if uploaded is not None:
        bytes_img = uploaded.getvalue()
        img_array = np.frombuffer(bytes_img, np.uint8)
        opencv_image = cv.imdecode(img_array, cv.IMREAD_COLOR)
        return opencv_image

def crop_img(img, pts):
    x_min, y_min =  pts[0]
    x_max, y_max =  pts[0]
# corners searching
    for pt in pts:
        x_min = pt[0] if pt[0]<x_min else x_min
        x_max = pt[0] if pt[0]>x_max else x_max
        y_min = pt[1] if pt[1]<y_min else y_min
        y_max = pt[1] if pt[1]>y_max else y_max
    # no se porque tengo que invertir el orden pero funciona
    return img[y_min:y_max, x_min:x_max ]

def rotacion(img, theta = 0, y=0):
    rows, cols, color = img.shape
    rotate_matrix = cv.getRotationMatrix2D(((cols-1)/2.0,y),theta,1)
    return cv.warpAffine(
    src=img, M=rotate_matrix, dsize=(cols, rows))
   
def gaussian_blur(img, k):
    return cv.GaussianBlur(img, (k,k), 0)

def apply_pipeline(img, filters):
    out = img.copy()
    for f in filters:
        out = f(out)
    return out

def canvas_to_pts(canvas, alpha):
    objects = pd.json_normalize(canvas.json_data["objects"])
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    
    if objects.get("path") is not None:
        polygon = eval(objects['path'][0])
        pts = []
        for pt in polygon:
            #st.write(pt)
            if len(pt)>1:
                pts.append([pt[1]/alpha,pt[2]/alpha]) 
        return pts
    else:
        return None

def quadratic_spline_roots(spl):
    roots = []
    knots = spl.get_knots()
    for a, b in zip(knots[:-1], knots[1:]):
        u, v, w = spl(a), spl((a+b)/2), spl(b)
        t = np.roots([u+w-2*v, w-u, 2*v])
        t = t[np.isreal(t) & (np.abs(t) <= 1)]
        roots.extend(t*(b-a)/2 + (b+a)/2)
    return np.array(roots)

st.set_page_config(layout='wide')
#st.set_page_config(page_title="Image Processing Pipeline", layout="wide")
# tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Upload & ROI", 
    "2️⃣ Preprocessing", 
    "3️⃣ Edging Detection", 
    "4️⃣ Analysis"
])

# Inicialización de estado de sesión al parecer es una clase de entorno a la cual podemos asignarle nuevos valores
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'roi_mask' not in st.session_state:
    st.session_state.roi_mask = None
if 'roi_image' not in st.session_state:
    st.session_state.roi_image = None
if 'filtered_image' not in st.session_state:
    st.session_state.filtered_image = None
if 'edges' not in st.session_state:
    st.session_state.edges = None


#========== ROI===========
with tab1:
    st.subheader("Select ROI")
    st.text("Right click for closing the polygon and saving it")

    uploaded = st.file_uploader("imagen para analizar", type=["jpg", "png"])

    # SELECCIÓN DE ROI
    # https://github.com/SunOner/streamlit-drawable-canvas

    alpha = 1/2 # factor de reduccion
    if uploaded is not None:

        img_rgb = load_img(uploaded)
        st.session_state.original_image = img_rgb
        img_pil = Image.fromarray(img_rgb)
        h,w = img_pil.size
        # el tamaño de la imagen importa ya que en términos del nuevo tamaño que pongamos estarán exportadas en el json las coordenadas de los vértices
        canvas = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            stroke_color="#ff0000",
            background_image=img_pil, 
            update_streamlit=True,
            height=alpha*w ,
            width=alpha*h ,
            drawing_mode="polygon",
            display_toolbar=True,
            key="canvas",
        )

        if canvas.json_data is not None:
            pts =  canvas_to_pts(canvas, alpha)
            if pts is not None:
                arr = [np.array(pts,'int')]
                #st.write(arr)
                #st.write(img_rgb.shape)
                #st.write(img_rgb.shape)
                st.session_state.roi_mask = cv.fillPoly(np.zeros(img_rgb.shape,np.uint8),arr,[1,1,1])
                st.session_state.roi_image = crop_img(np.multiply(img_rgb,st.session_state.roi_mask), arr[0])
                st.session_state.roi_mask = crop_img(st.session_state.roi_mask, arr[0])
                if st.session_state.roi_image is not None:
                    st.write("ROI mask succesfully saved")
                #st.image(img_roi_mask)
                # tenemos que cargar la imagen en un formato para que OpenCV pueda procesarla
# ========= Preprocessing ==========
with tab2: 
    st.header("Image Preprocessing")
    
    # necesito cambiar el filtro bilateral por uno gaussiano usual al aprecer, ya que necesito un gradiente para el cuál se arrastre la imagen. 
    # también necesito reponer el menu de rotación para además de alinear la imagen, seleccionar dos puntos con base en la recta.
    if st.session_state.roi_image is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            with st.container(height=400):  # Set fixed height in pixels
                st.subheader("Filter Controls")
                filters = [] 
                
#                apply_bilateral = st.checkbox("Apply Bilateral Filtering", value=True)
#                if apply_bilateral:
#                    radius = st.slider("Radius", 1, 100, 9, step=1)
#                    sigmaColor = st.slider("Sigma Color", 0, 100, 75, step=1)
#                    sigmaSpace = st.slider("Sigma Space", 0,100 , 75, step=1)
#                    filters.append(lambda img, radius=radius, sigmaColor = sigmaColor, sigmaSpace=sigmaSpace: cv.bilateralFilter(img, radius, sigmaColor, sigmaSpace))
#

                if st.checkbox("Rotación"):
                    rows = st.session_state.roi_image.shape[0]
                    theta = st.slider("Ángulo de rotación", -5.0, 5.0,0.01)
                    y = st.slider("Altura de pivote", 0, rows-1, 150)
                    filters.append(lambda img, theta=theta, y=y: rotacion(img, theta, y))
                else: 
                    y =0


                if st.checkbox("Difusión gaussiana"):
                    k = st.slider("Kernel size", 1, 31, 5, step=2)
                    filters.append(lambda img, k=k: gaussian_blur(img, k))

                apply_contrast = st.checkbox("Adjust Contrast", value=False)
                if apply_contrast:
                    alpha = st.slider("Contrast (alpha)", 0.5, 3.0, 1.0, 0.1)
                    beta = st.slider("Brightness (beta)", -100, 100, 0)
                    filters.append(lambda img, alpha=alpha, beta=beta:  cv.convertScaleAbs(src = img, alpha=alpha, beta=beta))

                
                #img = st.session_state.roi_image.copy()
                st.session_state.filtered_image = apply_pipeline(st.session_state.roi_image, filters)
               # if st.session_state.filtered_image is not None:
                #    st.success("✅ Filters applied!")
            
        with col2:
            st.subheader("Results")
            #col_a, col_b = st.columns(2)
            #with col_a:
            #st.image(st.session_state.roi_image, caption="Original ROI", use_container_width=True)
            #with col_b:
            if st.session_state.filtered_image is not None:
                fig, ax = plt.subplots(figsize=(16,9))
                im = ax.imshow(st.session_state.filtered_image, vmin=0, vmax=255)
                # Add initial horizontal line
                line = ax.axhline(y=y, color='red', linewidth=1)
                st.pyplot(fig)

                #st.image(st.session_state.filtered_image, caption="Filtered", use_container_width=True)
            else:
                st.info("Adjust filters and click 'Apply Filters'")
    else:
        st.warning("⚠️ Please select ROI in Tab 1 first")

#======= EDGING DETECTION =============
with tab3:
    st.subheader("Draw Initial Curve Approximation")

    if st.session_state.filtered_image is not None:
        img_edges = st.session_state.filtered_image.copy()[:,:,1]
        edge_image_pil = Image.fromarray(img_edges)

        #st.image(edge_image_pil)
        h2,w2 = edge_image_pil.size
        # st.write(h2,w2, img_edges.shape[0], img_edges.shape[1])
        # Use drawable canvas to get initial points
        alpha=1/2
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0.3)",
            stroke_width=2,
            stroke_color="#00FF00",
            background_image=edge_image_pil,
            update_streamlit=True,
            height=int(alpha*w2),
            width=int(alpha*h2),
            drawing_mode="polygon",  # or "polygon"
            display_toolbar=True,
            #key="snake_init_0",
        )
        if canvas_result.json_data is not None:
            pts =  canvas_to_pts(canvas_result, alpha)
            if pts is not None:
                st.success(f"✅ {len(pts)} points captured")

                # Parameter controls
                col1, col2 = st.columns(2)


                with col1:
                    alpha = st.slider("Elasticity (α)", 0.0, 1.0, 0.01, 0.001)
                    beta = st.slider("Rigidity (β)", -0.2, 0.2, 0.1, 0.001)
                    gamma = st.slider("Edge attraction (γ)", -2.0, 1.0, 0.1, 0.1)

                with col2:
                    num_iterations = st.slider("Iterations", 10, 500, 100)
                    num_points = st.slider("Interpolated points", 50, 300, 100)
                    window_size = st.slider("Window size", 1, 50, 5, 1)
                    threshold = 0.0001*st.slider("threshold", 0.01, 1.0, 0.1, 0.01)


                if st.button("Optimize Curve", type="primary"):
                    # Initialize snake from canvas points
                    snake = snk.initialize_snake_from_polygon(pts, num_points=100)
                    st.session_state.initial_snake = snake.copy()

                    optimized_snake, energy_history = snk.optimize_snake_greedy(
                        img_edges, snake,
                        num_iterations=num_iterations,
                        window_size=int(window_size),
                        alpha=alpha, beta=beta, gamma=gamma,
                        threshold=threshold
                    )
                    
                    # Store results
                    st.session_state.optimized_snake = optimized_snake
                    st.session_state.energy_history = energy_history

                    # Visualize
                    col_a, col_b = st.columns(2)

                    with col_a:
                        # Show initial
                        img_init = cv.cvtColor(img_edges.copy(), cv.COLOR_GRAY2RGB)
                        for i in range(len(pts)-1):
                            pt1 = tuple(int(x) for x in pts[i]) #tuple(snake[i].astype(int))
                            pt2 = tuple(int(x) for x in pts[i+1]) #tuple(snake[i+1].astype(int))
                            cv.line(img_init, pt1, pt2, (0, 255, 0), 2)
                        st.image(img_init, caption="Initial approximation")

                    with col_b:
                        # Show optimized
                        img_opt = cv.cvtColor(img_edges.copy(), cv.COLOR_GRAY2RGB)
                        for i in range(len(optimized_snake)-1):
                            pt1 = tuple(optimized_snake[i].astype(int))
                            pt2 = tuple(optimized_snake[i+1].astype(int))
                            cv.line(img_opt, pt1, pt2, (255, 0, 0), 2)
                        st.image(img_opt, caption="Optimized curve")

                    # Show coordinates
                    st.subheader("Extracted Curve Coordinates")
                    df = pd.DataFrame(optimized_snake, columns=['x', 'y'])
                    st.dataframe(df, height=200)

                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download coordinates",
                        csv,
                        "curve_coordinates.csv",
                        "text/csv"
                    )

    else:
        st.warning(" hoy no hay imagen ma~nana s'i")

with tab4:
    st.subheader("Curve analysis")
    
    df = pd.read_csv('curve_coordinates.csv')
    # st.write(df)
    x = df['x']
    y = max(df['y']) -  df['y']
    y_spl = UnivariateSpline(x,y,s=None,k=4)
    y_spl_1 = y_spl.derivative(1)
    y_spl_2 = y_spl.derivative(2)
    x_range = np.linspace(min(x),max(x),200)
    min_pt = y_spl_1.roots()
    critic_pts = quadratic_spline_roots(y_spl_2)

    R = abs(min_pt[1] - critic_pts[0])
    for pt in critic_pts:
        d = abs(min_pt[1] - pt)
        R = d if d<R else R
    radius_mask = lambda z: (z >= min_pt[1] - R) & (z <= min_pt[1] + R)

    mod_parab = modelo_parabolico(x[radius_mask(x)], y[radius_mask(x)])

    def model_branch(a,R, x):
        return [2*a*R**2 *(1 - R**2 / (2 *abs(xi)**2)) for xi in x]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Data points
    ax.scatter(x, y, s=30, alpha=0.6, color='#2E86AB', label='Data points', zorder=3)

    # Spline fit
    ax.plot(x_range, y_spl(x_range), linewidth=2, color='black', label='Spline fit', zorder=2)

    # First derivative (scaled for visibility)
    ax.plot(x_range, 100*y_spl_1(x_range) + 40, linewidth=1.5, 
            linestyle='--', color='gray', alpha=0.7, label="Spline derivative (×100, offset)")

    # Critical points
    ax.scatter(min_pt[1], y_spl(min_pt[1]), s=100, color='#A23B72', 
               marker='v', label='Minimum', zorder=4, edgecolors='black', linewidths=1.5)
    ax.scatter(critic_pts, y_spl(critic_pts), s=80, color='#F18F01', 
               marker='D', label='Critical points', zorder=4, edgecolors='black', linewidths=1)

    # Parabolic model
    x_parab = x_range[radius_mask(x_range)]
    ax.plot(x_parab, mod_parab.predict(x_parab), linewidth=2.5, 
            color='#C73E1D', linestyle='-.', label='Parabolic model', zorder=2)

    # Branches model
    x_branch = x_range[~radius_mask(x_range)]
    ax.plot(x_branch, model_branch(mod_parab._coeficientes[2], R, x_branch - min_pt[1] ), linewidth=2.5, 
            color='#C73E1D', linestyle='-.', label='Branch model', zorder=2)


    # Labels and styling
    ax.set_xlabel('x coordinate (pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('y coordinate (pixels)', fontsize=12, fontweight='bold')
    ax.set_title('Curve Fitting and Critical Point Analysis', fontsize=14, fontweight='bold', pad=15)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Legend
    ax.legend(loc='best', framealpha=0.9, fontsize=10, edgecolor='black')

    # Fixed axis limits (no autoscale)
    ax.set_xlim([x_range.min() - 5, x_range.max() + 5])
    ax.set_ylim([y.min() - 10, y.max() + 10])

    # Tight layout
    plt.tight_layout()

    st.pyplot(fig)
    #st.write(mod_parab._coeficientes )

